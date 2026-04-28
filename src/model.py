"""Random Forest model factory (exclusive model for this project)."""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(params: Dict[str, Any] | None = None) -> RandomForestClassifier:
    """Create RandomForestClassifier with required defaults."""
    base = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        base.update(params)
    return RandomForestClassifier(**base)


def hyperparameter_space() -> Dict[str, List[Any]]:
    """Lightweight hyperparameter search space."""
    return {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [4, 6, 8, 10, 12, 16],
        "model__min_samples_split": [5, 10, 20, 30],
        "model__min_samples_leaf": [2, 4, 8, 12],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__class_weight": ["balanced", "balanced_subsample"],
        "model__criterion": ["gini", "entropy", "log_loss"],
        "model__bootstrap": [True],
        "model__max_samples": [0.6, 0.8, 1.0],
        "model__ccp_alpha": [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    }
