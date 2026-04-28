"""Main dataset discovery, loading and profiling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DEFAULT_TARGET_COLUMN,
    MAIN_DATASET_PREFERRED_NAMES,
    TARGET_COLUMNS,
    resolve_target_column,
)
from .utils import hash_file, normalize_column_name, save_json


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}


def _find_dataset_file(data_dir: Path = DATA_DIR) -> Path:
    preferred = [data_dir / name for name in MAIN_DATASET_PREFERRED_NAMES]
    for file_path in preferred:
        if file_path.exists():
            return file_path

    candidates = sorted(
        p for p in data_dir.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(
            "No se encontró dataset principal en data/. "
            "Coloca hybrid_no_external_scores_dataset_ready(1).csv "
            "o hybrid_no_external_scores_dataset_ready.csv."
        )
    return candidates[0]


def _read_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    raise ValueError(f"Formato de dataset no soportado: {suffix}")


def _profile_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    bool_cols: List[str] = []
    text_cols: List[str] = []

    for col in df.columns:
        series = df[col]
        if is_bool_dtype(series):
            bool_cols.append(col)
        elif is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            # Distinguish short categorical from potentially textual columns.
            unique_non_null = series.dropna().astype(str).nunique()
            if unique_non_null <= 30:
                categorical_cols.append(col)
            else:
                text_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "boolean": bool_cols,
        "textual": text_cols,
    }


def _normalize_dataframe_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping = {col: normalize_column_name(col) for col in df.columns}
    normalized = df.rename(columns=mapping)
    return normalized, mapping


def _resolve_target(df: pd.DataFrame) -> Tuple[str, str, List[str]]:
    available = [c for c in df.columns if c in TARGET_COLUMNS or c.startswith("target_domain_")]
    selected, method = resolve_target_column(available_targets=available)
    if selected not in df.columns:
        raise ValueError(
            "No se encontró la columna objetivo seleccionada en el dataset. "
            f"TARGET_DISORDER={os.getenv('TARGET_DISORDER', '(vacío)')} "
            f"objetivo esperado={selected}"
        )
    return selected, method, available


def load_main_dataset(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and profile the principal dataset, then save artifacts/dataset_profile.json."""
    file_path = _find_dataset_file(data_dir=data_dir)
    raw_df = _read_table(file_path)
    df, original_to_normalized = _normalize_dataframe_columns(raw_df)
    column_profile = _profile_columns(df)
    target_col, target_method, available_targets = _resolve_target(df)

    target_dist = (
        df[target_col]
        .value_counts(dropna=False, normalize=False)
        .sort_index()
        .astype(int)
        .to_dict()
    )
    target_dist_pct = (
        df[target_col]
        .value_counts(dropna=False, normalize=True)
        .sort_index()
        .round(6)
        .to_dict()
    )

    warnings: List[str] = []
    if target_col != DEFAULT_TARGET_COLUMN:
        warnings.append(
            f"El target seleccionado no es el default de conducta: {target_col}."
        )
    if len(target_dist) <= 1:
        warnings.append(
            "El target tiene una sola clase. No se podrá entrenar un modelo binario válido."
        )

    profile = {
        "dataset_file_used": str(file_path.name),
        "dataset_file_path": str(file_path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_types_detected": {k: len(v) for k, v in column_profile.items()},
        "column_names_by_type": column_profile,
        "missing_values_by_column": df.isna().sum().to_dict(),
        "possible_targets": available_targets,
        "target_selected": target_col,
        "target_selection_method": target_method,
        "dataset_hash_sha256": hash_file(file_path),
        "target_distribution_counts": target_dist,
        "target_distribution_ratio": target_dist_pct,
        "warnings": warnings,
        "decision": (
            "Se seleccionó el target según TARGET_DISORDER o default conduct, "
            "manteniendo el dataset normalizado por nombres de columnas."
        ),
        "column_name_mapping_original_to_normalized": original_to_normalized,
    }
    save_json(ARTIFACTS_DIR / "dataset_profile.json", profile)
    return df, profile

