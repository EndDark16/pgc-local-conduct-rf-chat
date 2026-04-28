import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.predictor import ModelAssets, predict_with_assets
from src.preprocessing import build_preprocessor
from src.model import build_random_forest


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
        schema={},
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
