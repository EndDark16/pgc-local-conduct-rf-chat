import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.model import build_random_forest
from src.predictor import ModelAssets, answer_result_question, humanize_feature_name, predict_with_assets
from src.preprocessing import build_preprocessor


def test_predict_with_assets_mock():
    X = pd.DataFrame(
        {
            "age_years": [8, 9, 10, 11],
            "sex_assigned_at_birth": ["M", "F", "M", "F"],
            "conduct_02_initiates_fights": [0, 1, 2, 1],
        }
    )
    y = np.array([0, 0, 1, 1])
    preprocessor, _ = build_preprocessor(X)
    model = build_random_forest({"n_estimators": 30, "n_jobs": 1})
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X, y)

    assets = ModelAssets(
        model=pipe.named_steps["model"],
        preprocessor=pipe.named_steps["preprocessor"],
        metadata={
            "features_used": ["age_years", "sex_assigned_at_birth", "conduct_02_initiates_fights"],
            "thresholds": {"final": 0.55},
        },
        schema={
            "features": [
                {
                    "feature": "age_years",
                    "is_required": True,
                    "feature_label_human": "Edad",
                    "response_type": "numeric_range",
                    "min_value": 5,
                    "max_value": 18,
                },
                {
                    "feature": "sex_assigned_at_birth",
                    "is_required": True,
                    "feature_label_human": "Sexo asignado",
                    "response_type": "categorical",
                    "response_options": [{"value": "M", "label": "Masculino"}, {"value": "F", "label": "Femenino"}],
                },
                {
                    "feature": "conduct_02_initiates_fights",
                    "is_required": True,
                    "feature_label_human": "Inicia peleas",
                    "response_type": "temporal_0_2",
                    "response_options": [
                        {"value": 0, "label": "No ocurrió"},
                        {"value": 1, "label": "Ocurrió antes"},
                        {"value": 2, "label": "Ocurrió recientemente"},
                    ],
                },
            ],
            "feature_order": ["age_years", "sex_assigned_at_birth", "conduct_02_initiates_fights"],
        },
    )

    result = predict_with_assets(
        {
            "age_years": 12,
            "sex_assigned_at_birth": "M",
            "conduct_02_initiates_fights": 2,
        },
        assets=assets,
    )

    assert "probability_estimated" in result
    assert "threshold_used" in result
    assert "medical_disclaimer" in result
    assert "orientative_report" in result
    assert "compatibilidad" in result["orientative_report"]["compatibility_level"]
    assert result["orientative_report"]["title"] == "Impresión psicológica orientativa"
    assert "dictamen psicológico definitivo" in result["orientative_report"]["important_clarification"]
    indicators = result["orientative_report"]["observed_indicators"]
    assert all("conduct_" not in item for item in indicators)


def test_answer_result_question_local_rules():
    report = {
        "compatibility_level": "compatibilidad relevante",
        "observed_indicators": ["Inicia peleas: Ocurrió recientemente"],
        "professional_recommendation": "Se recomienda valoración profesional.",
        "technical_summary": {"threshold_used": 0.45},
    }
    out = answer_result_question(
        question="esto es un diagnostico",
        prediction_report=report,
        metrics={},
        feature_importance={},
        answers={"conduct_02_initiates_fights": 2},
    )
    assert "No" in out["answer"] or "no" in out["answer"]


def test_humanize_feature_name_no_technical_label():
    schema = {
        "features": [
            {
                "feature": "conduct_03_weapon_use",
                "caregiver_question": "¿Ha usado objetos o armas para causar daño?",
            }
        ]
    }
    label = humanize_feature_name("conduct_03_weapon_use", schema)
    assert "conduct_" not in label
    assert "_" not in label
