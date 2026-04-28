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


@dataclass
class ThresholdSummary:
    best_f1_threshold: float
    best_recall_threshold: float
    final_threshold: float
    rows: List[Dict[str, float]]


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

