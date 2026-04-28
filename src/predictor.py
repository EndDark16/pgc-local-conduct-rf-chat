"""Prediction utilities using trained Random Forest + preprocessor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, MEDICAL_DISCLAIMER, MODELS_DIR
from .utils import load_json


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


def load_model_assets() -> ModelAssets:
    model_path = MODELS_DIR / "model.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    metadata_path = MODELS_DIR / "metadata.json"
    schema_path = ARTIFACTS_DIR / "feature_schema.json"

    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(
            "Modelo no entrenado. Ejecuta primero `python train.py`."
        )
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    metadata = load_json(metadata_path, default={})
    schema = load_json(schema_path, default={})
    return ModelAssets(model=model, preprocessor=preprocessor, metadata=metadata, schema=schema)


def _ordered_input_frame(
    answers: Dict[str, Any],
    feature_order: List[str],
) -> pd.DataFrame:
    row = {feature: answers.get(feature, np.nan) for feature in feature_order}
    return pd.DataFrame([row])


def predict_with_assets(
    answers: Dict[str, Any],
    assets: ModelAssets,
) -> Dict[str, Any]:
    feature_order = assets.metadata.get("features_used") or assets.schema.get("feature_order") or []
    if not feature_order:
        raise ValueError("No se pudo resolver el esquema de features del modelo entrenado.")

    missing = [f for f in feature_order if f not in answers]
    required_features = [
        item.get("feature")
        for item in assets.schema.get("features", [])
        if bool(item.get("is_required", True))
    ]
    missing_required = [
        f for f in required_features if (f not in answers or answers.get(f) is None)
    ]
    if missing_required:
        raise ValueError(
            "Faltan respuestas obligatorias para generar la estimación: "
            + ", ".join(missing_required[:10])
        )

    X = _ordered_input_frame(answers, feature_order)
    X_t = assets.preprocessor.transform(X)
    probs = assets.model.predict_proba(X_t)
    probability = float(probs[:, 1][0]) if probs.shape[1] > 1 else float(probs[:, 0][0])

    threshold = float(
        assets.metadata.get("thresholds", {}).get("final", 0.5)
    )
    pred_class = int(probability >= threshold)
    compatible = "patrón compatible" if pred_class == 1 else "patrón no compatible"

    return {
        "probability_estimated": round(probability, 6),
        "threshold_used": threshold,
        "internal_class": pred_class,
        "compatibility_text": compatible,
        "risk_text": _risk_text(probability),
        "missing_features": missing,
        "explanation": (
            "La estimación se basa en respuestas interpretadas y confirmadas por el usuario. "
            "Este resultado expresa probabilidad estimada, no una conclusión clínica."
        ),
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }


def predict_from_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
    assets = load_model_assets()
    return predict_with_assets(answers=answers, assets=assets)
