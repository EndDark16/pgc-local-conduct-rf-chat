import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import MAX_ACCEPTABLE_METRIC
from src.model import build_random_forest
from src.preprocessing import build_preprocessor
from src.training_utils import evaluate_binary_metrics, select_model_with_overfit_guard, threshold_search


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


def test_max_acceptable_metric_constant():
    assert MAX_ACCEPTABLE_METRIC == 0.98


def test_select_model_with_overfit_guard_prefers_within_limit():
    candidates = [
        {
            "variant_name": "v1",
            "cv_selected_params": {"model__max_depth": 6, "model__min_samples_leaf": 8, "model__n_estimators": 150},
            "importance_concentration": 0.55,
            "train_metrics_final": {"f1": 0.99},
            "test_metrics_threshold_final": {
                "f1": 0.995,
                "recall": 0.995,
                "precision": 0.994,
                "accuracy": 0.995,
                "roc_auc": 0.995,
                "pr_auc": 0.995,
            },
        },
        {
            "variant_name": "v2",
            "cv_selected_params": {"model__max_depth": 4, "model__min_samples_leaf": 12, "model__n_estimators": 120},
            "importance_concentration": 0.28,
            "train_metrics_final": {"f1": 0.965},
            "test_metrics_threshold_final": {
                "f1": 0.962,
                "recall": 0.968,
                "precision": 0.95,
                "accuracy": 0.961,
                "roc_auc": 0.97,
                "pr_auc": 0.968,
            },
        },
    ]
    selected, report = select_model_with_overfit_guard(candidates)
    assert selected["variant_name"] == "v2"
    assert report["overfit_guard_applied"] is True
    assert report["overfit_warning"] is False
