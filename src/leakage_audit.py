"""Leakage auditing utilities for conservative model training."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR
from .utils import normalize_text, save_json


LEAKAGE_NAME_PATTERNS = [
    "target_domain_",
    "_count",
    "_score",
    "_sum",
    "_index",
    "_flag",
    "_final",
    "_total",
    "derived",
    "aggregate",
    "composite",
]

SAFE_EXCEPTIONS = {
    "age_years",
    "sex_assigned_at_birth",
}


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return series.astype("category").cat.codes.replace(-1, np.nan).astype(float)


def _is_name_suspicious(feature: str) -> Tuple[bool, str]:
    norm = normalize_text(feature)
    for pattern in LEAKAGE_NAME_PATTERNS:
        if pattern in norm:
            return True, f"name_pattern:{pattern}"
    return False, ""


def _value_is_near_deterministic(feature_series: pd.Series, y: pd.Series) -> Tuple[bool, Dict[str, Any]]:
    df = pd.DataFrame({"x": feature_series, "y": y}).dropna()
    if len(df) < 60:
        return False, {"reason": "insufficient_rows"}

    grouped = df.groupby("x")["y"].agg(["mean", "count"]).reset_index()
    if grouped.empty:
        return False, {"reason": "empty_group"}

    dominant = grouped[grouped["count"] >= 20]
    if dominant.empty:
        return False, {"reason": "small_groups"}

    strong = dominant[(dominant["mean"] <= 0.02) | (dominant["mean"] >= 0.98)]
    coverage = float(strong["count"].sum() / max(1, len(df)))
    if coverage >= 0.70 and len(strong) >= 2:
        return True, {
            "coverage": coverage,
            "strong_groups": int(len(strong)),
            "groups_total": int(len(grouped)),
        }
    return False, {
        "coverage": coverage,
        "strong_groups": int(len(strong)),
        "groups_total": int(len(grouped)),
    }


def run_leakage_audit(
    dataset_df: pd.DataFrame,
    y_binary: pd.Series,
    candidate_features: List[str],
    target_col: str,
    persist: bool = True,
) -> Dict[str, Any]:
    reviewed: List[Dict[str, Any]] = []
    high_correlation: List[Dict[str, Any]] = []
    suspicious: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []

    y_num = pd.to_numeric(y_binary, errors="coerce")
    for feature in candidate_features:
        if feature not in dataset_df.columns:
            excluded.append({"feature": feature, "reason": "missing_in_dataset"})
            continue

        name_suspicious, reason = _is_name_suspicious(feature)
        col = dataset_df[feature]
        col_num = _coerce_numeric(col)
        corr = None
        if col_num.notna().sum() > 5:
            try:
                corr_val = col_num.corr(y_num)
                corr = None if pd.isna(corr_val) else float(corr_val)
            except Exception:
                corr = None

        deterministic, deterministic_detail = _value_is_near_deterministic(col, y_num)

        feature_flags: List[str] = []
        if name_suspicious:
            feature_flags.append(reason)
        if corr is not None and abs(corr) >= 0.90:
            high_correlation.append({"feature": feature, "abs_correlation": abs(corr)})
            feature_flags.append("high_correlation")
        if deterministic:
            feature_flags.append("near_deterministic_separation")

        reviewed.append(
            {
                "feature": feature,
                "name_suspicious": name_suspicious,
                "correlation_with_target": corr,
                "deterministic_like": deterministic,
                "deterministic_detail": deterministic_detail,
                "flags": feature_flags,
            }
        )

        if feature_flags:
            suspicious.append(
                {
                    "feature": feature,
                    "flags": feature_flags,
                    "correlation_with_target": corr,
                    "deterministic_detail": deterministic_detail,
                }
            )

        if name_suspicious and feature not in SAFE_EXCEPTIONS:
            excluded.append({"feature": feature, "reason": reason})

    high_correlation.sort(key=lambda x: float(x["abs_correlation"]), reverse=True)

    audit = {
        "target_column": target_col,
        "reviewed_count": len(reviewed),
        "reviewed_columns": reviewed,
        "high_correlation_features": high_correlation,
        "suspicious_features": suspicious,
        "columns_excluded": excluded,
        "summary": {
            "excluded_count": len(excluded),
            "high_correlation_count": len(high_correlation),
            "suspicious_count": len(suspicious),
        },
        "decision": (
            "Se excluyen automáticamente las variables con patrones de nombre típicos de fuga. "
            "Las variables altamente correlacionadas o casi determinísticas se marcan para variantes conservadoras."
        ),
    }

    if persist:
        save_json(ARTIFACTS_DIR / "leakage_audit.json", audit)
    return audit

