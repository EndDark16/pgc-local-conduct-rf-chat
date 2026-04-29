"""FastAPI local web app with robust conversational flow and result QA."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from .audit import audit_event
from .config import ARTIFACTS_DIR, MEDICAL_DISCLAIMER, MODELS_DIR, WEB_STATIC_DIR, WEB_TEMPLATES_DIR
from .data_loader import load_main_dataset
from .feature_selection import DEFAULT_CONDUCT_FEATURES, is_feature_allowed_for_target, select_features
from .nlp_interpreter import interpret_answer, is_help_request
from .predictor import answer_result_question, predict_from_answers
from .preprocessing import build_feature_schema
from .question_explainer import explain_question
from .question_generator import question_for_feature
from .questionnaire_loader import load_questionnaire
from .response_options import format_response_options
from .utils import as_bool, load_json, normalize_text


app = FastAPI(title="PGC Local Conduct Estimator", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(WEB_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(WEB_TEMPLATES_DIR))


TECH_TEXT_PATTERNS = [
    r"\binput for\b",
    r"\bresponse_options_json\b",
    r"\bfeature_name\b",
    r"\btarget_domain_\w+\b",
    r"\b[a-z]+_[a-z0-9_]+\b",
]

TARGET_INTROS = {
    "target_domain_conduct_final": "Te haré preguntas sobre comportamientos observables relacionados con normas, convivencia y conducta.",
    "target_domain_adhd_final": "Te haré preguntas sobre atención, inquietud e impulsividad.",
    "target_domain_anxiety_final": "Te haré preguntas sobre preocupaciones, miedos o señales de ansiedad.",
    "target_domain_depression_final": "Te haré preguntas sobre estado de ánimo, interés y energía.",
    "target_domain_elimination_final": "Te haré preguntas relacionadas con control de esfínteres y situaciones asociadas.",
}


@dataclass
class SessionState:
    role: str = "caregiver"
    answers_confirmed: Dict[str, Any] = field(default_factory=dict)
    attempts_by_feature: Dict[str, int] = field(default_factory=dict)
    last_interpretation_by_feature: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    latest_prediction: Dict[str, Any] = field(default_factory=dict)


SESSIONS: Dict[str, SessionState] = {}


class ChatInterpretRequest(BaseModel):
    feature: Optional[str] = None
    feature_name: Optional[str] = None
    answer: str
    role: str = "caregiver"
    session_id: str = "default"
    question_metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatExplainRequest(BaseModel):
    feature: Optional[str] = None
    feature_name: Optional[str] = None
    mode: str = "simple"


class ConfirmRequest(BaseModel):
    feature: Optional[str] = None
    feature_name: Optional[str] = None
    parsed_value: Any
    raw_answer: str = ""
    confidence: float = 0.0
    session_id: str = "default"
    used_missing_strategy: bool = False


class PredictRequest(BaseModel):
    session_id: str = "default"
    answers: Optional[Dict[str, Any]] = None


class ResetRequest(BaseModel):
    session_id: str = "default"


class ResultQuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


class AuditRequest(BaseModel):
    event: str
    payload: Dict[str, Any] = Field(default_factory=dict)


def _get_session(session_id: str) -> SessionState:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = SessionState()
    return SESSIONS[session_id]


def _get_feature_name(payload: Any) -> str:
    feature = getattr(payload, "feature", None) or getattr(payload, "feature_name", None)
    return str(feature or "").strip()


def _sanitize_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _sanitize_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_jsonable(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _looks_technical_text(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    for pattern in TECH_TEXT_PATTERNS:
        if re.search(pattern, norm):
            return True
    return False


def _safe_user_text(*candidates: Any, default_text: str = "") -> str:
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and not _looks_technical_text(text):
            return text
    return default_text


def _resolve_target_for_runtime() -> str:
    metadata = load_json(MODELS_DIR / "metadata.json", default={})
    if metadata.get("target_column"):
        return str(metadata["target_column"])
    dataset_profile = load_json(ARTIFACTS_DIR / "dataset_profile.json", default={})
    return str(dataset_profile.get("target_selected") or "target_domain_conduct_final")


def _schema_domain_stats(schema: Dict[str, Any]) -> Dict[str, int]:
    stats = {"adhd": 0, "conduct": 0, "other": 0}
    for item in schema.get("features", []):
        feature = str(item.get("feature", ""))
        if feature.startswith("adhd_"):
            stats["adhd"] += 1
        elif feature.startswith("conduct_") or feature in {"age_years", "sex_assigned_at_birth"}:
            stats["conduct"] += 1
        else:
            stats["other"] += 1
    return stats


def _schema_is_inconsistent(schema: Dict[str, Any], target_col: str) -> bool:
    if not schema.get("features"):
        return True
    stats = _schema_domain_stats(schema)
    if target_col == "target_domain_conduct_final":
        if stats["adhd"] > 0:
            return True
        if stats["conduct"] == 0:
            return True
    return False


def _build_schema_from_contract() -> Dict[str, Any]:
    questionnaire_df, _, feature_meta = load_questionnaire()
    dataset_df, dataset_profile = load_main_dataset()
    target_col = str(dataset_profile.get("target_selected") or "target_domain_conduct_final")

    selected_report = select_features(dataset_df, questionnaire_df, target_col=target_col)
    selected = [f for f in selected_report.get("features_used", []) if f in feature_meta]
    if not selected:
        selected = [f for f in DEFAULT_CONDUCT_FEATURES if f in dataset_df.columns and f in feature_meta]
    if not selected:
        raise RuntimeError(
            "No se encontraron preguntas para el target seleccionado. "
            "Revisa artifacts/feature_schema.json o ejecuta python train.py."
        )

    schema = build_feature_schema(selected_features=selected, feature_to_metadata=feature_meta, persist=True)
    audit_event(
        "schema_regenerated_from_contract",
        {
            "target_column": target_col,
            "features_count": len(selected),
        },
    )
    return schema


def _load_schema() -> Dict[str, Any]:
    schema_path = ARTIFACTS_DIR / "feature_schema.json"
    schema = load_json(schema_path, default={})
    target_col = _resolve_target_for_runtime()

    if not schema.get("features"):
        schema = _build_schema_from_contract()

    if _schema_is_inconsistent(schema, target_col):
        audit_event(
            "schema_inconsistent_detected",
            {
                "target_column": target_col,
                "schema_stats": _schema_domain_stats(schema),
                "message": "Se detectó incoherencia entre target y schema. Se intentará regenerar.",
            },
        )
        schema = _build_schema_from_contract()
        if _schema_is_inconsistent(schema, target_col):
            raise RuntimeError(
                "Se detectaron artefactos incoherentes para el target seleccionado. "
                "Ejecuta python train.py para regenerar modelo y esquema."
            )
    return _sanitize_jsonable(schema)


def _schema_map(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item["feature"]: item for item in schema.get("features", []) if item.get("feature")}


def _normalize_role(role: str) -> str:
    role_norm = str(role or "caregiver").strip().lower()
    if role_norm in {"psychologist", "psicologo", "psicólogo"}:
        return "psychologist"
    return "caregiver"


def _question_allowed(meta: Dict[str, Any], target_col: str) -> bool:
    feature = str(meta.get("feature") or "").strip()
    if not feature:
        return False
    allowed, _ = is_feature_allowed_for_target(feature, meta, target_col)
    return allowed


def _build_question_payload(meta: Dict[str, Any], role: str, idx: int, total: int, target_col: str) -> Dict[str, Any]:
    generated = question_for_feature(meta, role=role)
    explanation = explain_question(str(meta.get("feature")), {**meta, **generated})

    question_text = _safe_user_text(
        generated.get("question"),
        meta.get("question_text_primary"),
        default_text="¿Puedes contarme qué has observado en esta situación?",
    )
    help_text = _safe_user_text(generated.get("help_text"), explanation.get("simple_explanation"))
    scale_guidance = _safe_user_text(generated.get("scale_guidance"))

    options_payload = format_response_options(
        response_options=generated.get("response_options") or meta.get("response_options"),
        response_options_json=meta.get("response_options_json"),
        response_type=meta.get("response_type"),
        scale_guidance=scale_guidance,
        help_text=help_text,
        question=question_text,
        min_value=meta.get("min_value"),
        max_value=meta.get("max_value"),
    )

    return _sanitize_jsonable(
        {
            "feature": str(meta.get("feature")),
            "question": question_text,
            "help_text": help_text,
            "scale_guidance": scale_guidance,
            "response_options": options_payload["options_list"],
            "response_type": meta.get("response_type"),
            "min_value": meta.get("min_value"),
            "max_value": meta.get("max_value"),
            "feature_label_human": _safe_user_text(meta.get("feature_label_human"), default_text=""),
            "term_explanation": _safe_user_text(meta.get("term_explanation"), default_text=""),
            "examples": explanation.get("examples") or [],
            "simple_explanation": explanation.get("simple_explanation") or "",
            "human_options_text": options_payload["human_options_text"],
            "quick_chips": options_payload["quick_chips"],
            "scale_type": options_payload["scale_type"],
            "progress_index": idx,
            "progress_total": total,
            "target_column": target_col,
            "is_required": bool(as_bool(meta.get("is_required", True)) if "is_required" in meta else True),
        }
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"medical_disclaimer": MEDICAL_DISCLAIMER})


@app.get("/api/model-status")
async def model_status() -> Dict[str, Any]:
    model_exists = (MODELS_DIR / "model.joblib").exists() and (MODELS_DIR / "preprocessor.joblib").exists()
    metadata = load_json(MODELS_DIR / "metadata.json", default={})
    target_column = str(metadata.get("target_column") or "target_domain_conduct_final")
    return {
        "model_trained": model_exists,
        "target_column": target_column,
        "target_intro": TARGET_INTROS.get(target_column, TARGET_INTROS["target_domain_conduct_final"]),
        "threshold_final": metadata.get("thresholds", {}).get("final"),
        "features_count": metadata.get("n_features_used"),
        "medical_disclaimer": MEDICAL_DISCLAIMER,
        "message": "Modelo listo para predicción." if model_exists else "Modelo no entrenado. Ejecuta python train.py",
    }


@app.get("/api/questions")
async def get_questions(role: str = Query(default="caregiver"), session_id: str = Query(default="default")) -> Dict[str, Any]:
    role_resolved = _normalize_role(role)
    target_col = _resolve_target_for_runtime()
    schema = _load_schema()
    raw_features = schema.get("features", [])

    filtered = [item for item in raw_features if _question_allowed(item, target_col)]
    if not filtered:
        raise HTTPException(
            status_code=500,
            detail=(
                "No se encontraron preguntas para el target seleccionado. "
                "Revisa artifacts/feature_schema.json o ejecuta python train.py."
            ),
        )

    questions = [
        _build_question_payload(meta, role=role_resolved, idx=index + 1, total=len(filtered), target_col=target_col)
        for index, meta in enumerate(filtered)
    ]

    if target_col == "target_domain_conduct_final":
        leaked = [q["feature"] for q in questions if str(q.get("feature", "")).startswith("adhd_")]
        if leaked:
            raise HTTPException(
                status_code=500,
                detail="Error de validación: se detectaron preguntas ADHD en target de conducta.",
            )

    session = _get_session(session_id)
    session.role = role_resolved
    session.latest_prediction = {}

    audit_event(
        "chat_questions_loaded",
        {
            "role": role_resolved,
            "target_column": target_col,
            "count": len(questions),
            "session_id": session_id,
            "features": [q.get("feature") for q in questions],
        },
    )

    return {
        "role": role_resolved,
        "target_column": target_col,
        "intro_text": TARGET_INTROS.get(target_col, TARGET_INTROS["target_domain_conduct_final"]),
        "total": len(questions),
        "questions": questions,
    }


@app.post("/api/chat/explain")
async def api_chat_explain(payload: ChatExplainRequest) -> Dict[str, Any]:
    feature = _get_feature_name(payload)
    if not feature:
        raise HTTPException(status_code=400, detail="Feature no especificada.")

    schema = _load_schema()
    meta = _schema_map(schema).get(feature)
    if not meta:
        raise HTTPException(status_code=404, detail="No se encontró la pregunta solicitada.")

    explanation = explain_question(feature, meta)
    audit_event("chat_question_explained", {"feature": feature, "mode": payload.mode})
    return _sanitize_jsonable(explanation)


@app.post("/api/chat/interpret")
async def api_chat_interpret(payload: ChatInterpretRequest) -> Dict[str, Any]:
    feature = _get_feature_name(payload)
    if not feature:
        raise HTTPException(status_code=400, detail="Feature no especificada para interpretar.")

    schema = _load_schema()
    meta = _schema_map(schema).get(feature)
    if not meta:
        raise HTTPException(status_code=404, detail="No se encontró la pregunta para interpretar.")

    merged_meta = {**meta, **(payload.question_metadata or {})}

    session = _get_session(payload.session_id)
    session.role = _normalize_role(payload.role)
    attempt = session.attempts_by_feature.get(feature, 0) + 1
    session.attempts_by_feature[feature] = attempt

    if is_help_request(payload.answer):
        explanation = explain_question(feature, merged_meta)
        audit_event(
            "chat_user_needs_explanation",
            {
                "feature": feature,
                "raw_answer": payload.answer,
                "attempt": attempt,
                "session_id": payload.session_id,
            },
        )
        return {
            "ok": True,
            "interpreted": None,
            "needs_explanation": True,
            "explanation": _sanitize_jsonable(explanation),
            "message": "Te explico la pregunta en palabras más simples.",
        }

    interpreted = interpret_answer(feature_name=feature, raw_answer=payload.answer, metadata=merged_meta, attempt=attempt, max_attempts=3)
    interpreted["is_required"] = bool(as_bool(merged_meta.get("is_required", True)))
    interpreted["attempt"] = attempt
    interpreted["allow_missing_value"] = bool(
        interpreted.get("needs_clarification") and attempt >= 3 and not interpreted["is_required"]
    )
    interpreted["max_attempts_reached"] = bool(attempt >= 3)

    session.last_interpretation_by_feature[feature] = interpreted

    audit_event(
        "chat_answer_interpreted",
        {
            "feature": feature,
            "raw_answer": payload.answer,
            "parsed_value": interpreted.get("parsed_value"),
            "confidence": interpreted.get("confidence"),
            "category": interpreted.get("answer_category"),
            "attempt": attempt,
            "session_id": payload.session_id,
            "needs_clarification": interpreted.get("needs_clarification"),
            "scale_type": interpreted.get("scale_type"),
        },
    )
    return {"ok": True, "interpreted": _sanitize_jsonable(interpreted), "needs_explanation": False}


@app.post("/api/chat/confirm")
async def api_chat_confirm(payload: ConfirmRequest) -> Dict[str, Any]:
    feature = _get_feature_name(payload)
    if not feature:
        raise HTTPException(status_code=400, detail="Feature no especificada para confirmar.")

    session = _get_session(payload.session_id)
    last = session.last_interpretation_by_feature.get(feature)
    if not last and payload.parsed_value is None:
        raise HTTPException(status_code=400, detail="No hay una interpretación previa para confirmar.")

    schema = _load_schema()
    meta = _schema_map(schema).get(feature, {})
    is_required = bool(as_bool(meta.get("is_required", True)) if "is_required" in meta else True)

    if payload.parsed_value is None and is_required:
        raise HTTPException(
            status_code=400,
            detail="Este dato es importante para hacer la estimación. Necesito una respuesta más clara.",
        )

    parsed_value = payload.parsed_value
    if parsed_value is None and last:
        parsed_value = last.get("parsed_value")

    if parsed_value is None and is_required:
        raise HTTPException(status_code=400, detail="No se puede confirmar sin un valor válido.")

    session.answers_confirmed[feature] = parsed_value
    session.attempts_by_feature[feature] = 0

    audit_event(
        "chat_answer_confirmed",
        {
            "feature": feature,
            "parsed_value": parsed_value,
            "raw_answer": payload.raw_answer,
            "confidence": payload.confidence,
            "session_id": payload.session_id,
            "is_required": is_required,
        },
    )

    return {"ok": True, "confirmed_answers_count": len(session.answers_confirmed)}


@app.post("/api/predict")
async def api_predict(payload: PredictRequest) -> Dict[str, Any]:
    session = _get_session(payload.session_id)
    answers = payload.answers or session.answers_confirmed
    if not answers:
        raise HTTPException(status_code=400, detail="Aún no hay respuestas confirmadas.")

    try:
        prediction = predict_from_answers(answers)
    except FileNotFoundError:
        return {
            "ok": False,
            "message": "Modelo no entrenado. Ejecuta python train.py",
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        }
    except ValueError as value_error:
        return {
            "ok": False,
            "message": f"Faltan algunos datos importantes para la estimación. {value_error}",
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        }

    session.latest_prediction = prediction

    audit_event(
        "chat_prediction_generated",
        {
            "session_id": payload.session_id,
            "probability_estimated": prediction.get("probability_estimated"),
            "threshold_used": prediction.get("threshold_used"),
            "internal_class": prediction.get("internal_class"),
            "answers_count": len(answers),
            "overfit_warning": prediction.get("overfit_warning"),
        },
    )

    return {"ok": True, "prediction": _sanitize_jsonable(prediction)}


@app.post("/api/chat/result-question")
async def api_chat_result_question(payload: ResultQuestionRequest) -> Dict[str, Any]:
    session = _get_session(payload.session_id)
    if not session.latest_prediction:
        raise HTTPException(status_code=400, detail="Aún no hay resultado final para explicar.")

    metrics = load_json(ARTIFACTS_DIR / "metrics.json", default={})
    feature_importance = load_json(ARTIFACTS_DIR / "feature_importance.json", default={})
    report = session.latest_prediction.get("orientative_report", {})

    answer_payload = answer_result_question(
        question=payload.question,
        prediction_report=report,
        metrics=metrics,
        feature_importance=feature_importance,
        answers=session.answers_confirmed,
    )

    audit_event(
        "chat_result_question_answered",
        {
            "session_id": payload.session_id,
            "question": payload.question,
            "answer": answer_payload.get("answer"),
        },
    )

    return {
        "ok": True,
        "answer": answer_payload.get("answer", ""),
        "chips": session.latest_prediction.get("result_qa_chips", []),
    }


@app.post("/api/reset-session")
async def api_reset_session(payload: ResetRequest) -> Dict[str, Any]:
    SESSIONS[payload.session_id] = SessionState()
    audit_event("chat_session_reset", {"session_id": payload.session_id})
    return {"ok": True, "message": "Sesión reiniciada"}


@app.get("/api/metrics")
async def api_metrics() -> Dict[str, Any]:
    metrics = load_json(ARTIFACTS_DIR / "metrics.json", default={})
    if not metrics:
        return {"ok": False, "message": "No existen métricas aún. Ejecuta python train.py"}

    warning = None
    test_final = metrics.get("test_metrics_threshold_final", {})
    if any(float(test_final.get(k) or 0.0) > 0.98 for k in ["accuracy", "precision", "recall", "f1"]):
        warning = (
            "Las métricas son muy altas. Esto puede indicar que el dataset contiene señales muy directas "
            "del target o que se requiere validación externa adicional."
        )

    return {
        "ok": True,
        "target_column": metrics.get("target_column"),
        "features_count": metrics.get("features_count"),
        "threshold_0_5": _sanitize_jsonable(metrics.get("test_metrics_threshold_0_5", {})),
        "threshold_final": _sanitize_jsonable(test_final),
        "confusion_matrix": _sanitize_jsonable(test_final.get("confusion_matrix")),
        "overfit_warning": warning,
    }


@app.get("/api/feature-importance")
async def api_feature_importance() -> Dict[str, Any]:
    data = load_json(ARTIFACTS_DIR / "feature_importance.json", default={})
    if not data:
        return {"ok": False, "message": "No existe importancia de variables aún. Ejecuta python train.py"}
    return {
        "ok": True,
        "feature_importance_aggregated": _sanitize_jsonable(data.get("feature_importance_aggregated", [])[:20]),
        "note": "Estas variables fueron relevantes para el modelo, pero no significan causa directa.",
    }


@app.post("/api/audit-event")
async def api_audit_event(payload: AuditRequest) -> Dict[str, Any]:
    audit_event(payload.event, payload.payload)
    return {"ok": True}


# Backward-compatible aliases.
@app.post("/api/explain-question")
async def api_explain_question_legacy(payload: ChatExplainRequest) -> Dict[str, Any]:
    return await api_chat_explain(payload)


@app.post("/api/interpret")
async def api_interpret_legacy(payload: ChatInterpretRequest) -> Dict[str, Any]:
    return await api_chat_interpret(payload)


@app.post("/api/confirm-response")
async def api_confirm_response_legacy(payload: ConfirmRequest) -> Dict[str, Any]:
    return await api_chat_confirm(payload)


@app.post("/api/confirm-answer")
async def api_confirm_answer_alias(payload: ConfirmRequest) -> Dict[str, Any]:
    return await api_chat_confirm(payload)
