"""Random Forest model factory (exclusive model for this project)."""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(params: Dict[str, Any] | None = None) -> RandomForestClassifier:
    """Create RandomForestClassifier with required defaults."""
    base = {
        "n_estimators": 150,
        "max_depth": 6,
        "min_samples_split": 30,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
        "max_samples": 0.7,
        "bootstrap": True,
        "ccp_alpha": 0.001,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        base.update(params)
    return RandomForestClassifier(**base)


def hyperparameter_space() -> Dict[str, List[Any]]:
    """Regularized hyperparameter search space focused on generalization."""
    return {
        "model__n_estimators": [100, 150, 200],
        "model__max_depth": [3, 4, 5, 6],
        "model__min_samples_split": [20, 30, 40, 60],
        "model__min_samples_leaf": [5, 8, 10, 15],
        "model__max_features": ["sqrt", "log2", 0.4, 0.5],
        "model__class_weight": ["balanced", "balanced_subsample"],
        "model__criterion": ["gini", "entropy"],
        "model__bootstrap": [True],
        "model__max_samples": [0.5, 0.6, 0.7, 0.8],
        "model__ccp_alpha": [0.0, 0.001, 0.002, 0.005, 0.01],
    }
