"""Prediction utilities and orientative reporting for the local chat application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, MEDICAL_DISCLAIMER, MODELS_DIR
from .response_options import format_response_options
from .utils import load_json, normalize_text


@dataclass
class ModelAssets:
    model: Any
    preprocessor: Any
    metadata: Dict[str, Any]
    schema: Dict[str, Any]


def _risk_text(probability: float) -> str:
    if probability < 0.35:
        return "riesgo estimado bajo"
    if probability < 0.65:
        return "riesgo estimado intermedio"
    return "riesgo estimado alto"


def _compatibility_level(probability: float, threshold: float) -> str:
    low_cut = max(0.10, threshold * 0.75)
    high_cut = min(0.90, threshold + 0.20)

    if probability < low_cut:
        return "compatibilidad baja"
    if probability < threshold:
        return "compatibilidad intermedia o zona de observacion"
    if probability < high_cut:
        return "compatibilidad relevante"
    return "compatibilidad alta"


def _schema_map(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("feature")): item
        for item in schema.get("features", [])
        if str(item.get("feature", "")).strip()
    }


def _feature_label(meta: Dict[str, Any], feature: str) -> str:
    label = str(meta.get("feature_label_human") or "").strip()
    if label:
        return label
    question = str(meta.get("caregiver_question") or meta.get("question_text_primary") or "").strip()
    if question:
        return question
    return feature


def _human_value(meta: Dict[str, Any], value: Any) -> str:
    options_payload = format_response_options(
        response_options=meta.get("response_options"),
        response_options_json=meta.get("response_options_json"),
        response_type=meta.get("response_type"),
        scale_guidance=meta.get("scale_guidance"),
        help_text=meta.get("help_text"),
        question=meta.get("question_text_primary") or meta.get("caregiver_question"),
        min_value=meta.get("min_value"),
        max_value=meta.get("max_value"),
    )

    for option in options_payload["options_list"]:
        if option.get("value") == value:
            return str(option.get("label") or value)

    scale_type = options_payload.get("scale_type")
    if scale_type == "binary":
        return "Si" if int(value) == 1 else "No"
    if scale_type == "temporal_0_2":
        mapping = {0: "No ocurrio", 1: "Ocurrio antes", 2: "Ocurrio recientemente"}
        return mapping.get(int(value), str(value))
    if scale_type == "frequency_0_3":
        mapping = {0: "Nunca", 1: "Ocasional", 2: "Frecuente", 3: "Casi siempre"}
        return mapping.get(int(value), str(value))
    if scale_type == "observation_0_2":
        mapping = {0: "No se observa", 1: "A veces", 2: "Claramente"}
        return mapping.get(int(value), str(value))
    if scale_type == "impact_0_3":
        mapping = {0: "Sin impacto", 1: "Leve", 2: "Moderado", 3: "Marcado"}
        return mapping.get(int(value), str(value))
    return str(value)


def _top_indicators(
    answers: Dict[str, Any],
    schema: Dict[str, Any],
    feature_importance: Optional[Dict[str, Any]] = None,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    schema_by_feature = _schema_map(schema)
    importance_map: Dict[str, float] = {}
    if feature_importance:
        for row in feature_importance.get("feature_importance_aggregated", []):
            feature = str(row.get("feature") or "")
            if feature:
                importance_map[feature] = float(row.get("importance") or 0.0)

    rows: List[Dict[str, Any]] = []
    for feature, value in answers.items():
        if value is None:
            continue
        meta = schema_by_feature.get(feature, {})
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = 0.0

        if numeric_value <= 0:
            continue

        rows.append(
            {
                "feature": feature,
                "label": _feature_label(meta, feature),
                "value": value,
                "value_text": _human_value(meta, value),
                "importance": importance_map.get(feature, 0.0),
                "score": (importance_map.get(feature, 0.0) * 2.0) + numeric_value,
            }
        )

    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[:top_k]


def build_orientative_psychological_report(
    prediction: Dict[str, Any],
    answers: Dict[str, Any],
    schema: Dict[str, Any],
    feature_importance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    probability = float(prediction.get("probability_estimated", 0.0))
    threshold = float(prediction.get("threshold_used", 0.5))
    compatibility_level = _compatibility_level(probability, threshold)

    indicators = _top_indicators(answers, schema, feature_importance=feature_importance, top_k=6)
    indicators_text = [f"{row['label']}: {row['value_text']}" for row in indicators]

    if compatibility_level == "compatibilidad baja":
        summary = (
            "La impresion orientativa muestra baja compatibilidad con el dominio evaluado. "
            "Esto sugiere que, con las respuestas registradas, no predominan indicadores conductuales de alta intensidad."
        )
    elif compatibility_level == "compatibilidad intermedia o zona de observacion":
        summary = (
            "La impresion orientativa se ubica en zona intermedia. "
            "Hay algunas senales conductuales que conviene observar con seguimiento estructurado."
        )
    elif compatibility_level == "compatibilidad relevante":
        summary = (
            "La impresion orientativa muestra compatibilidad relevante con el dominio evaluado. "
            "Se observan indicadores conductuales consistentes que ameritan valoracion profesional."
        )
    else:
        summary = (
            "La impresion orientativa muestra compatibilidad alta con el dominio evaluado. "
            "Las respuestas describen un patron conductual de alta presencia e impacto potencial."
        )

    functional_impact = (
        "La informacion sugiere que el comportamiento podria afectar convivencia, normas o funcionamiento cotidiano. "
        "Es recomendable contrastar estos hallazgos con entrevistas y observacion clinica."
    )

    recommendation = (
        "Se recomienda una valoracion por psicologia clinica o neuropsicologia infantil, "
        "incluyendo entrevista con cuidadores y contexto escolar, para confirmar o descartar hipotesis diagnosticas."
    )

    technical_summary = {
        "probability_estimated": probability,
        "threshold_used": threshold,
        "compatibility_level": compatibility_level,
        "risk_text": prediction.get("risk_text"),
        "internal_class": prediction.get("internal_class"),
    }

    return {
        "title": "Impresion orientativa del dominio evaluado",
        "compatibility_level": compatibility_level,
        "clinical_style_summary": summary,
        "observed_indicators": indicators_text,
        "functional_impact": functional_impact,
        "professional_recommendation": recommendation,
        "important_clarification": (
            "Esta interpretacion es una estimacion preliminar generada por un sistema de apoyo. "
            "No constituye un dictamen psicologico definitivo, no reemplaza una valoracion clinica "
            "y no debe usarse como diagnostico medico o psicologico preciso."
        ),
        "suggested_questions": [
            "¿Que situaciones concretas activan estas conductas?",
            "¿Con que frecuencia ocurren en casa y escuela?",
            "¿Que estrategias de manejo se han intentado y con que resultado?",
        ],
        "technical_summary": technical_summary,
    }


def answer_result_question(
    question: str,
    prediction_report: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    feature_importance: Optional[Dict[str, Any]],
    answers: Dict[str, Any],
) -> Dict[str, Any]:
    q = normalize_text(question)
    technical = prediction_report.get("technical_summary", {})

    if any(token in q for token in ["que significa", "significa este resultado", "compatibilidad"]):
        return {
            "answer": (
                f"El resultado indica {prediction_report.get('compatibility_level', 'un nivel de compatibilidad')}. "
                "Compatibilidad significa que las respuestas se parecen, en mayor o menor medida, "
                "al patron que el modelo aprendio para este dominio."
            )
        }

    if any(token in q for token in ["por que", "porque", "por que salio", "por que dio", "variables", "indicadores"]):
        indicators = prediction_report.get("observed_indicators", [])
        if indicators:
            joined = "; ".join(indicators[:5])
            return {
                "answer": (
                    "El resultado se apoyo principalmente en estos indicadores observados: "
                    f"{joined}. "
                    "Esto no implica causalidad directa, solo relevancia estadistica para el modelo."
                )
            }
        return {
            "answer": "El sistema considera el conjunto de respuestas confirmadas y su patron global para estimar compatibilidad."
        }

    if any(token in q for token in ["que debo hacer", "que sigue", "recomendacion", "ahora que"]):
        return {"answer": prediction_report.get("professional_recommendation", "Se recomienda valoracion profesional calificada.")}

    if any(token in q for token in ["diagnostico", "es un diagnostico", "dictamen"]):
        return {
            "answer": (
                "No. Este resultado no es un diagnostico. Es una impresion orientativa automatizada "
                "que debe ser revisada por un profesional calificado."
            )
        }

    if "threshold" in q:
        threshold = technical.get("threshold_used")
        return {
            "answer": (
                f"El threshold es el punto de corte de decision del modelo. En esta evaluacion fue {threshold}. "
                "Si la probabilidad supera ese valor, el modelo marca mayor compatibilidad."
            )
        }

    if any(token in q for token in ["recall", "f1", "precision", "metrica", "metricas"]):
        if not metrics:
            return {"answer": "Aun no hay metricas disponibles en esta sesion. Puedes ejecutar entrenamiento y revisar el panel tecnico."}
        final_m = metrics.get("test_metrics_threshold_final", {})
        return {
            "answer": (
                "En el conjunto de prueba, el modelo se evalua con F1 para balance general y recall para no pasar por alto casos relevantes. "
                f"Valores actuales: F1={final_m.get('f1')}, Recall={final_m.get('recall')}, Precision={final_m.get('precision')}."
            )
        }

    if any(token in q for token in ["repetir", "nueva evaluacion", "reiniciar"]):
        return {"answer": "Puedes usar el boton 'Iniciar nueva evaluacion' para reiniciar desde cero con una sesion limpia."}

    return {
        "answer": (
            "Puedo ayudarte a explicar el nivel de compatibilidad, los indicadores principales, "
            "las metricas del modelo o los pasos recomendados."
        )
    }


def load_model_assets() -> ModelAssets:
    model_path = MODELS_DIR / "model.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    metadata_path = MODELS_DIR / "metadata.json"
    schema_path = ARTIFACTS_DIR / "feature_schema.json"

    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError("Modelo no entrenado. Ejecuta primero `python train.py`.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    metadata = load_json(metadata_path, default={})
    schema = load_json(schema_path, default={})
    return ModelAssets(model=model, preprocessor=preprocessor, metadata=metadata, schema=schema)


def _ordered_input_frame(answers: Dict[str, Any], feature_order: List[str]) -> pd.DataFrame:
    row = {feature: answers.get(feature, np.nan) for feature in feature_order}
    return pd.DataFrame([row])


def _overfit_alert(metrics: Dict[str, Any]) -> Optional[str]:
    if not metrics:
        return None
    final = metrics.get("test_metrics_threshold_final", {})
    train_final = metrics.get("train_metrics_final", {})
    candidates = [
        float(final.get("accuracy") or 0.0),
        float(final.get("precision") or 0.0),
        float(final.get("recall") or 0.0),
        float(final.get("f1") or 0.0),
    ]
    max_metric = max(candidates) if candidates else 0.0
    gap = max(0.0, float(train_final.get("f1") or 0.0) - float(final.get("f1") or 0.0))

    if max_metric > 0.98:
        return (
            "Las metricas son muy altas. Esto puede indicar que el dataset contiene senales muy directas "
            "del target o que se requiere validacion externa adicional."
        )
    if gap > 0.08:
        return (
            "Se detecta una brecha relevante entre entrenamiento y prueba. "
            "Conviene revisar regularizacion y validacion externa."
        )
    return None


def predict_with_assets(answers: Dict[str, Any], assets: ModelAssets) -> Dict[str, Any]:
    feature_order = assets.metadata.get("features_used") or assets.schema.get("feature_order") or []
    if not feature_order:
        raise ValueError("No se pudo resolver el esquema de features del modelo entrenado.")

    required_features = [
        item.get("feature")
        for item in assets.schema.get("features", [])
        if bool(item.get("is_required", True))
    ]
    missing_required = [f for f in required_features if f not in answers or answers.get(f) is None]
    if missing_required:
        raise ValueError(
            "Faltan respuestas obligatorias para generar la estimacion: " + ", ".join(missing_required[:10])
        )

    X = _ordered_input_frame(answers, feature_order)
    X_t = assets.preprocessor.transform(X)
    probs = assets.model.predict_proba(X_t)
    probability = float(probs[:, 1][0]) if probs.shape[1] > 1 else float(probs[:, 0][0])

    threshold = float(assets.metadata.get("thresholds", {}).get("final", 0.5))
    pred_class = int(probability >= threshold)
    compatibility_text = "patron compatible" if pred_class == 1 else "patron no compatible"

    metrics = load_json(ARTIFACTS_DIR / "metrics.json", default={})
    feature_importance = load_json(ARTIFACTS_DIR / "feature_importance.json", default={})

    base_prediction = {
        "probability_estimated": round(probability, 6),
        "threshold_used": threshold,
        "internal_class": pred_class,
        "compatibility_text": compatibility_text,
        "risk_text": _risk_text(probability),
        "missing_features": [f for f in feature_order if f not in answers],
        "explanation": (
            "La estimacion se basa en respuestas interpretadas y confirmadas por el usuario. "
            "Este resultado expresa probabilidad estimada, no una conclusion clinica."
        ),
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }

    report = build_orientative_psychological_report(
        prediction=base_prediction,
        answers=answers,
        schema=assets.schema,
        feature_importance=feature_importance,
    )

    return {
        **base_prediction,
        "orientative_report": report,
        "result_qa_chips": [
            "¿Que significa este resultado?",
            "¿Por que salio asi?",
            "¿Que debo hacer ahora?",
            "¿Esto es un diagnostico?",
            "Ver indicadores principales",
            "Repetir encuesta",
        ],
        "overfit_warning": _overfit_alert(metrics),
        "metrics_snapshot": {
            "f1": metrics.get("test_metrics_threshold_final", {}).get("f1"),
            "recall": metrics.get("test_metrics_threshold_final", {}).get("recall"),
            "precision": metrics.get("test_metrics_threshold_final", {}).get("precision"),
            "accuracy": metrics.get("test_metrics_threshold_final", {}).get("accuracy"),
        },
    }


def predict_from_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
    assets = load_model_assets()
    return predict_with_assets(answers=answers, assets=assets)
