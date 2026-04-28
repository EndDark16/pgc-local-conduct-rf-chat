"""Questionnaire dictionary loader and profiler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import ARTIFACTS_DIR, DATA_DIR, QUESTIONNAIRE_PREFERRED_NAMES
from .utils import hash_file, normalize_column_name, save_json, try_parse_json


MIN_REQUIRED_COLUMNS = [
    "feature",
    "question_text_primary",
    "caregiver_question",
    "psychologist_question",
    "feature_label_human",
    "feature_description",
    "term_explanation",
    "section_name",
    "subsection_name",
    "domains_final",
    "feature_type",
    "feature_role",
    "response_type",
    "response_options_json",
    "min_value",
    "max_value",
    "unit",
    "scale_guidance",
    "help_text",
    "who_answers",
    "respondent_expected",
    "administered_by",
    "visible_question_yes_no",
    "show_in_questionnaire_yes_no",
    "requires_clinician_administration",
    "requires_child_self_report",
    "question_audit_status",
]


def _find_questionnaire_file(data_dir: Path = DATA_DIR) -> Path:
    for name in QUESTIONNAIRE_PREFERRED_NAMES:
        path = data_dir / name
        if path.exists():
            return path
    candidates = sorted(data_dir.glob("questionnaire*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "No se encontró el CSV de preguntas humanizadas en data/. "
        "Se esperaba questionnaire_v16_4_visible_questions_excel_utf8.csv."
    )


def _parse_options_column(df: pd.DataFrame) -> pd.DataFrame:
    if "response_options_json" not in df.columns:
        return df
    df = df.copy()
    df["response_options"] = df["response_options_json"].apply(try_parse_json)
    return df


def load_questionnaire(
    data_dir: Path = DATA_DIR,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Load questionnaire metadata and save artifacts/questionnaire_profile.json."""
    file_path = _find_questionnaire_file(data_dir=data_dir)
    df = pd.read_csv(file_path)
    source_rows, source_cols = df.shape
    original_columns = df.columns.tolist()

    # Keep original columns but ensure feature keys are normalized for matching.
    if "feature" not in df.columns:
        raise ValueError(
            "El CSV de preguntas no contiene la columna 'feature'. "
            "No es posible construir el contrato de entradas."
        )
    df["feature_raw"] = df["feature"].astype(str)
    df["feature"] = df["feature"].astype(str).map(normalize_column_name)
    df = _parse_options_column(df)

    warnings: List[str] = []
    missing_min_columns = [c for c in MIN_REQUIRED_COLUMNS if c not in original_columns]
    if missing_min_columns:
        warnings.append(
            "Faltan columnas mínimas en cuestionario: " + ", ".join(missing_min_columns)
        )

    feature_to_metadata: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        feature = row.get("feature")
        if not feature:
            continue
        feature_to_metadata[str(feature)] = row.to_dict()

    profile = {
        "questionnaire_file_used": str(file_path.name),
        "questionnaire_file_path": str(file_path),
        "rows": int(source_rows),
        "columns": int(source_cols),
        "source_shape": [int(source_rows), int(source_cols)],
        "processed_shape": [int(df.shape[0]), int(df.shape[1])],
        "column_names": df.columns.tolist(),
        "source_column_names": original_columns,
        "missing_min_required_columns": missing_min_columns,
        "warnings": warnings,
        "questionnaire_hash_sha256": hash_file(file_path),
        "features_count": int(df["feature"].nunique()),
        "decision": (
            "Se cargó el diccionario de preguntas como contrato oficial de inputs. "
            "Se normalizó feature para alinear con columnas del dataset."
        ),
    }
    save_json(ARTIFACTS_DIR / "questionnaire_profile.json", profile)
    return df, profile, feature_to_metadata
