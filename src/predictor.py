"""Prediction utilities and orientative reporting for the local chat application."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, MAX_ACCEPTABLE_METRIC, MEDICAL_DISCLAIMER, MODELS_DIR
from .response_options import format_response_options
from .training_utils import metrics_above_limit
from .utils import load_json, normalize_text


TECH_PATTERNS = [
    r"\b[a-z]+_[a-z0-9_]+\b",
    r"\binput for\b",
    r"\btarget_domain_\w+\b",
]

CONDUCT_FALLBACK_LABELS = {
    "conduct_impairment_global": "afectación global en la vida diaria",
    "conduct_onset_before_10": "inicio temprano de conductas problemáticas (antes de los 10 años)",
    "conduct_01_bullies_threatens_intimidates": "acoso o intimidación a otras personas",
    "conduct_02_initiates_fights": "inicio de peleas físicas",
    "conduct_03_weapon_use": "uso de objetos o armas que podrían causar daño grave",
    "conduct_04_physical_cruelty_people": "crueldad física hacia otras personas",
    "conduct_05_physical_cruelty_animals": "crueldad física hacia animales",
    "conduct_06_steals_confronting_victim": "robo con amenaza, fuerza o intimidación",
    "conduct_07_forced_sex": "conducta sexual forzada",
    "conduct_08_fire_setting": "provocar incendios de manera deliberada",
    "conduct_09_property_destruction": "destrucción deliberada de propiedad",
    "conduct_10_breaks_into_house_building_car": "entrar por la fuerza a casa, edificio o vehículo",
    "conduct_11_lies_to_obtain_or_avoid": "mentiras repetidas para obtener beneficios o evitar consecuencias",
    "conduct_12_steals_without_confrontation": "robo sin confrontación directa",
    "conduct_13_stays_out_at_night_before_13": "permanecer fuera de casa por la noche antes de los 13 años",
    "conduct_14_runs_away_overnight": "escaparse de casa durante la noche",
    "conduct_15_truancy_before_13": "ausencias injustificadas a la escuela antes de los 13 años",
    "conduct_lpe_01_lack_remorse_guilt": "dificultad para mostrar remordimiento o culpa",
    "conduct_lpe_02_callous_lack_empathy": "baja empatía hacia otras personas",
    "conduct_lpe_03_unconcerned_performance": "poca preocupación por el desempeño o las consecuencias",
    "conduct_lpe_04_shallow_deficient_affect": "expresión emocional superficial o limitada",
    "age_years": "edad del niño",
    "sex_assigned_at_birth": "sexo asignado al nacer",
}


@dataclass
class ModelAssets:
    model: Any
    preprocessor: Any
    metadata: Dict[str, Any]
    schema: Dict[str, Any]


def _looks_technical(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    for pattern in TECH_PATTERNS:
        if re.search(pattern, norm):
            return True
    return False


def humanize_feature_name(feature: str, schema: Dict[str, Any]) -> str:
    feature = str(feature or "").strip()
    schema_map = {
        str(item.get("feature")): item
        for item in schema.get("features", [])
        if str(item.get("feature", "")).strip()
    }
    meta = schema_map.get(feature, {})

    question = str(meta.get("caregiver_question") or meta.get("question_text_primary") or "").strip()
    if question and not _looks_technical(question):
        q = question.strip("¿?").strip()
        return q[:1].lower() + q[1:] if q else question

    label = str(meta.get("feature_label_human") or "").strip()
    if label and not _looks_technical(label):
        return label

    description = str(meta.get("feature_description") or "").strip()
    if description and not _looks_technical(description):
        return description

    if feature in CONDUCT_FALLBACK_LABELS:
        return CONDUCT_FALLBACK_LABELS[feature]

    clean = re.sub(r"^(conduct|adhd|anxiety|depression|elimination)_", "", feature)
    clean = re.sub(r"^\d+_", "", clean)
    clean = clean.replace("_", " ").strip()
    clean = re.sub(r"\s+", " ", clean)
    return clean if clean else "indicador observado"


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
        return "compatibilidad intermedia o zona de observación"
    if probability < high_cut:
        return "compatibilidad relevante"
    return "compatibilidad alta"


def _schema_map(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("feature")): item
        for item in schema.get("features", [])
        if str(item.get("feature", "")).strip()
    }


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
        return "Sí" if int(value) == 1 else "No"
    if scale_type == "temporal_0_2":
        mapping = {0: "No ocurrió", 1: "Ocurrió antes", 2: "Ocurrió recientemente"}
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
                "label": humanize_feature_name(feature, schema),
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
        synthesis = (
            "La impresión orientativa sugiere baja compatibilidad con el dominio evaluado. "
            "En las respuestas registradas no predominan indicadores conductuales de alta intensidad."
        )
    elif compatibility_level == "compatibilidad intermedia o zona de observación":
        synthesis = (
            "La impresión orientativa se ubica en una zona intermedia. "
            "Se observan algunas señales que conviene monitorear con seguimiento estructurado."
        )
    elif compatibility_level == "compatibilidad relevante":
        synthesis = (
            "La impresión orientativa muestra compatibilidad relevante con el dominio evaluado. "
            "Existen indicadores consistentes que justifican una valoración profesional."
        )
    else:
        synthesis = (
            "La impresión orientativa muestra compatibilidad alta con el dominio evaluado. "
            "Las respuestas describen un patrón conductual de alta presencia y posible impacto funcional."
        )

    functional_impact = (
        "La información sugiere impacto potencial en convivencia, cumplimiento de normas, "
        "dinámica familiar y funcionamiento escolar. Es recomendable contrastar estos hallazgos "
        "con entrevistas y observación clínica."
    )

    recommendation = (
        "Se recomienda valoración por psicología clínica o neuropsicología infantil, "
        "entrevista con cuidadores, contraste con contexto escolar y aplicación de instrumentos formales."
    )

    technical_summary = {
        "probability_estimated": probability,
        "threshold_used": threshold,
        "compatibility_level": compatibility_level,
        "risk_text": prediction.get("risk_text"),
        "internal_class": prediction.get("internal_class"),
    }

    return {
        "title": "Impresión psicológica orientativa",
        "general_synthesis": synthesis,
        "compatibility_level": compatibility_level,
        "observed_indicators": indicators_text,
        "functional_impact": functional_impact,
        "professional_recommendation": recommendation,
        "important_clarification": (
            "Esta interpretación es una estimación preliminar generada por un sistema de apoyo. "
            "No constituye un dictamen psicológico definitivo, no reemplaza una valoración clínica "
            "y no debe usarse como diagnóstico médico o psicológico preciso."
        ),
        "suggested_questions": [
            "¿Qué situaciones concretas activan estas conductas?",
            "¿Con qué frecuencia ocurren en casa y escuela?",
            "¿Qué estrategias de manejo se han intentado y con qué resultado?",
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
    indicators = prediction_report.get("observed_indicators", [])
    level = prediction_report.get("compatibility_level", "compatibilidad no disponible")

    if any(token in q for token in ["que significa", "significa este resultado", "compatibilidad"]):
        return {
            "answer": (
                f"Este resultado indica {level}. "
                "Compatibilidad significa qué tan parecidas son las respuestas al patrón que el modelo aprendió "
                "para este dominio, pero no equivale a un diagnóstico clínico."
            )
        }

    if any(token in q for token in ["por que", "porque", "por que salio", "por que dio", "indicadores", "variables"]):
        if indicators:
            joined = "; ".join(indicators[:5])
            return {
                "answer": (
                    f"El resultado se apoyó principalmente en estos indicadores: {joined}. "
                    "Estas variables son relevantes para el modelo, pero no implican causalidad directa."
                )
            }
        return {"answer": "El resultado se basa en el patrón global de respuestas confirmadas durante la evaluación."}

    if any(token in q for token in ["que debo hacer", "que sigue", "recomendacion", "ahora que"]):
        if "relevante" in level or "alta" in level:
            return {
                "answer": (
                    prediction_report.get("professional_recommendation", "")
                    + " Dado el nivel observado, es importante priorizar una evaluación profesional pronta."
                )
            }
        return {"answer": prediction_report.get("professional_recommendation", "Se recomienda valoración profesional.")}

    if any(token in q for token in ["diagnostico", "es un diagnostico", "dictamen"]):
        return {
            "answer": (
                "No. Esta salida no es un diagnóstico. Es una impresión orientativa automatizada que sirve como apoyo "
                "y siempre debe ser revisada por un profesional calificado."
            )
        }

    if any(token in q for token in ["ver indicadores", "indicadores principales"]):
        if indicators:
            return {"answer": "Indicadores principales observados: " + "; ".join(indicators[:6])}
        return {"answer": "No hay indicadores destacados para esta sesión."}

    if "threshold" in q:
        threshold = technical.get("threshold_used")
        return {
            "answer": (
                f"El threshold es el punto de corte del modelo. En esta evaluación fue {threshold}. "
                "Cuando la probabilidad supera ese valor, el sistema marca mayor compatibilidad."
            )
        }

    if any(token in q for token in ["recall", "f1", "precision", "metrica", "metricas"]):
        if not metrics:
            return {"answer": "Aún no hay métricas disponibles en esta sesión."}
        final_m = metrics.get("test_metrics_threshold_final", {})
        return {
            "answer": (
                "El modelo se evalúa priorizando F1 y recall. "
                f"Valores actuales: F1={final_m.get('f1')}, Recall={final_m.get('recall')}, "
                f"Precision={final_m.get('precision')}."
            )
        }

    if any(token in q for token in ["repetir", "nueva evaluacion", "reiniciar"]):
        return {"answer": "Puedes usar el botón 'Iniciar nueva evaluación' para reiniciar la sesión desde cero."}

    return {
        "answer": (
            "Puedo ayudarte a explicar el nivel de compatibilidad, los indicadores que influyeron, "
            "las métricas del modelo o los pasos recomendados."
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


def _overfit_alert(metrics: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
    final = metrics.get("test_metrics_threshold_final", {})
    above = metrics_above_limit(final, limit=MAX_ACCEPTABLE_METRIC)
    if above:
        return (
            "Las métricas son muy altas. Esto puede indicar que el dataset contiene señales muy directas del target "
            "o que se requiere validación externa adicional."
        )
    if bool(metadata.get("overfit_warning", False)):
        return (
            "El modelo fue marcado con advertencia de sobreajuste potencial tras aplicar controles de regularización. "
            "Se recomienda validación externa."
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
            "Faltan respuestas obligatorias para generar la estimación: " + ", ".join(missing_required[:10])
        )

    X = _ordered_input_frame(answers, feature_order)
    X_t = assets.preprocessor.transform(X)
    probs = assets.model.predict_proba(X_t)
    probability = float(probs[:, 1][0]) if probs.shape[1] > 1 else float(probs[:, 0][0])

    threshold = float(assets.metadata.get("thresholds", {}).get("final", 0.5))
    pred_class = int(probability >= threshold)
    compatibility_text = "patrón compatible" if pred_class == 1 else "patrón no compatible"

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
            "La estimación se basa en respuestas interpretadas y confirmadas por el usuario. "
            "Este resultado expresa probabilidad estimada, no una conclusión clínica."
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
            "¿Qué significa este resultado?",
            "¿Por qué salió así?",
            "¿Qué debo hacer ahora?",
            "¿Esto es un diagnóstico?",
            "Ver indicadores principales",
            "Repetir encuesta",
        ],
        "overfit_warning": _overfit_alert(metrics, assets.metadata),
        "metrics_snapshot": {
            "f1": metrics.get("test_metrics_threshold_final", {}).get("f1"),
            "recall": metrics.get("test_metrics_threshold_final", {}).get("recall"),
            "precision": metrics.get("test_metrics_threshold_final", {}).get("precision"),
            "accuracy": metrics.get("test_metrics_threshold_final", {}).get("accuracy"),
            "roc_auc": metrics.get("test_metrics_threshold_final", {}).get("roc_auc"),
            "pr_auc": metrics.get("test_metrics_threshold_final", {}).get("pr_auc"),
        },
    }


def predict_from_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
    assets = load_model_assets()
    return predict_with_assets(answers=answers, assets=assets)
