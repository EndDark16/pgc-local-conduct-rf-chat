"""Training and threshold helper functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import MAX_ACCEPTABLE_METRIC


@dataclass
class ThresholdSummary:
    best_f1_threshold: float
    best_recall_threshold: float
    final_threshold: float
    rows: List[Dict[str, float]]


METRIC_KEYS_PRIMARY = ["f1", "recall", "precision", "accuracy", "roc_auc", "pr_auc"]


def evaluate_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    out: Dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = None
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        out["pr_auc"] = None
    return out


def threshold_search(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    max_acceptable_metric: float = MAX_ACCEPTABLE_METRIC,
    conservative_recall_ceiling: float | None = None,
) -> ThresholdSummary:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.901, 0.01)

    rows: List[Dict[str, float]] = []
    best_f1 = (-1.0, 0.5, 0.0)  # f1, threshold, recall
    best_recall = (-1.0, 0.5, 0.0)  # recall, threshold, f1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append(
            {
                "threshold": float(round(threshold, 4)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

        if f1 > best_f1[0] or (f1 == best_f1[0] and recall > best_f1[2]):
            best_f1 = (f1, threshold, recall)

        if recall > best_recall[0] or (recall == best_recall[0] and f1 > best_recall[2]):
            best_recall = (recall, threshold, f1)

    recall_ceiling = (
        float(conservative_recall_ceiling)
        if conservative_recall_ceiling is not None
        else min(float(max_acceptable_metric), 0.95)
    )

    guarded_candidates = [
        r
        for r in rows
        if r["recall"] <= recall_ceiling
        and r["precision"] <= float(max_acceptable_metric)
        and r["f1"] <= float(max_acceptable_metric)
    ]
    if guarded_candidates:
        final = max(guarded_candidates, key=lambda r: (r["f1"], r["recall"], r["precision"]))
        final_threshold = final["threshold"]
    else:
        max_f1 = best_f1[0]
        candidates = [r for r in rows if r["f1"] >= max_f1 * 0.95]
        if candidates:
            final = max(candidates, key=lambda r: (r["recall"], r["f1"]))
            final_threshold = final["threshold"]
        else:
            final_threshold = float(best_f1[1])

    return ThresholdSummary(
        best_f1_threshold=float(round(best_f1[1], 4)),
        best_recall_threshold=float(round(best_recall[1], 4)),
        final_threshold=float(round(final_threshold, 4)),
        rows=rows,
    )


def metrics_above_limit(
    metrics: Dict[str, Any],
    limit: float = MAX_ACCEPTABLE_METRIC,
) -> List[Dict[str, float]]:
    above: List[Dict[str, float]] = []
    for key in METRIC_KEYS_PRIMARY:
        value = metrics.get(key)
        if value is None:
            continue
        value_float = float(value)
        if value_float > limit:
            above.append({"metric": key, "value": value_float})
    return above


def _model_complexity_score(params: Dict[str, Any]) -> float:
    depth = params.get("model__max_depth")
    split = params.get("model__min_samples_split", 2)
    leaf = params.get("model__min_samples_leaf", 1)
    estimators = params.get("model__n_estimators", 100)
    max_samples = params.get("model__max_samples", 1.0)
    alpha = params.get("model__ccp_alpha", 0.0)

    depth_num = 12.0 if depth is None else float(depth)
    return (
        depth_num * 0.2
        + float(estimators) * 0.002
        + max(0.0, 30.0 - float(split)) * 0.02
        + max(0.0, 10.0 - float(leaf)) * 0.03
        + (1.0 - float(max_samples)) * 0.5
        + max(0.0, 0.01 - float(alpha)) * 10.0
    )


def select_model_with_overfit_guard(
    results: List[Dict[str, Any]],
    max_acceptable_metric: float = MAX_ACCEPTABLE_METRIC,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not results:
        raise ValueError("No hay resultados para seleccionar modelo con overfit guard.")

    evaluated: List[Dict[str, Any]] = []
    for item in results:
        test_metrics = item.get("test_metrics_threshold_final", {})
        train_metrics = item.get("train_metrics_final", {})
        params = item.get("cv_selected_params", {})
        overfit_gap = max(
            0.0,
            float(train_metrics.get("f1") or 0.0) - float(test_metrics.get("f1") or 0.0),
        )
        above = metrics_above_limit(test_metrics, limit=max_acceptable_metric)
        overflow_sum = sum(float(x["value"]) - max_acceptable_metric for x in above)
        importance_concentration = float(item.get("importance_concentration", 0.0))
        complexity_score = _model_complexity_score(params)
        within_limit = len(above) == 0

        if within_limit:
            guard_score = (
                float(test_metrics.get("f1") or 0.0) * 1.5
                + float(test_metrics.get("recall") or 0.0)
                + float(test_metrics.get("precision") or 0.0) * 0.35
                - overfit_gap * 0.9
                - importance_concentration * 0.3
                - complexity_score * 0.02
            )
        else:
            guard_score = (
                float(test_metrics.get("f1") or 0.0)
                + float(test_metrics.get("recall") or 0.0) * 0.2
                - overflow_sum * 4.5
                - overfit_gap * 0.9
                - importance_concentration * 0.3
                - complexity_score * 0.03
            )

        evaluated.append(
            {
                **item,
                "within_limit": within_limit,
                "metrics_above_limit": above,
                "overflow_sum": float(overflow_sum),
                "overfit_gap_train_test_f1": float(overfit_gap),
                "complexity_score": float(complexity_score),
                "guard_score": float(guard_score),
            }
        )

    within = [x for x in evaluated if x["within_limit"]]
    if within:
        chosen = max(
            within,
            key=lambda x: (
                float(x["guard_score"]),
                float(x.get("test_metrics_threshold_final", {}).get("f1") or 0.0),
                float(x.get("test_metrics_threshold_final", {}).get("recall") or 0.0),
                -float(x["complexity_score"]),
            ),
        )
        overfit_warning = False
        reason = (
            "Se seleccionó la variante con mejor F1/recall dentro del límite máximo aceptable "
            "y con brecha de generalización controlada."
        )
    else:
        chosen = max(
            evaluated,
            key=lambda x: (
                -float(x["overflow_sum"]),
                float(x["guard_score"]),
                -float(x["complexity_score"]),
            ),
        )
        overfit_warning = True
        reason = (
            "No hubo variantes que cumplieran el límite de 0.98. "
            "Se eligió la opción más conservadora disponible y se marca validación externa obligatoria."
        )

    report = {
        "max_acceptable_metric": float(max_acceptable_metric),
        "overfit_guard_applied": True,
        "overfit_warning": overfit_warning,
        "selected_model_variant": chosen.get("variant_name"),
        "selected_model_reason": reason,
        "selected_metrics_above_limit": chosen.get("metrics_above_limit", []),
        "evaluated_variants": [
            {
                "variant_name": v.get("variant_name"),
                "guard_score": v.get("guard_score"),
                "within_limit": v.get("within_limit"),
                "metrics_above_limit": v.get("metrics_above_limit", []),
                "overflow_sum": v.get("overflow_sum"),
                "overfit_gap_train_test_f1": v.get("overfit_gap_train_test_f1"),
                "importance_concentration": v.get("importance_concentration"),
                "complexity_score": v.get("complexity_score"),
            }
            for v in sorted(evaluated, key=lambda e: float(e["guard_score"]), reverse=True)
        ],
    }
    return chosen, report
