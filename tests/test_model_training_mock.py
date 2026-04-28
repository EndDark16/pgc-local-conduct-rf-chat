import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.model import build_random_forest
from src.preprocessing import build_preprocessor
from src.training_utils import evaluate_binary_metrics, threshold_search


def test_mock_training_random_forest():
    X = pd.DataFrame(
        {
            "age_years": [8, 9, 10, 11, 12, 13, 14, 15],
            "sex_assigned_at_birth": ["M", "F", "M", "F", "M", "F", "M", "F"],
            "conduct_02_initiates_fights": [0, 0, 1, 1, 2, 2, 2, 1],
        }
    )
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    preprocessor, _ = build_preprocessor(X)
    model = build_random_forest({"n_estimators": 50, "random_state": 42, "n_jobs": 1})
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X, y)
    probs = pipe.predict_proba(X)[:, 1]
    summary = threshold_search(y, probs)
    assert 0.1 <= summary.best_f1_threshold <= 0.9
    metrics = evaluate_binary_metrics(y, probs, threshold=summary.final_threshold)
    assert "f1" in metrics and metrics["f1"] >= 0.0
