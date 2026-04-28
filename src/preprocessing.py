"""Preprocessing pipeline and feature schema generation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import ARTIFACTS_DIR, MODELS_DIR
from .response_options import format_response_options
from .utils import as_bool, safe_float, save_json


SCHEMA_FIELDS = [
    "feature",
    "questionnaire_item_id",
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
    "response_options",
    "min_value",
    "max_value",
    "unit",
    "scale_guidance",
    "help_text",
    "who_answers",
    "respondent_expected",
    "administered_by",
    "requires_clinician_administration",
    "requires_child_self_report",
    "question_audit_status",
    "is_required",
    "parser_rules",
    "fallback_question",
]


def prepare_features_frame(df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """Ensure selected features exist and have stable order."""
    out = pd.DataFrame(index=df.index)
    for feature in selected_features:
        if feature in df.columns:
            out[feature] = df[feature]
        else:
            out[feature] = pd.NA
    return out


def infer_feature_groups(
    X_df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    numeric_cols: List[str] = []
    bool_cols: List[str] = []
    categorical_cols: List[str] = []
    excluded_text_cols: List[str] = []

    for col in X_df.columns:
        series = X_df[col]
        if is_bool_dtype(series):
            bool_cols.append(col)
            continue
        if is_numeric_dtype(series):
            numeric_cols.append(col)
            continue
        non_null = series.dropna().astype(str)
        avg_len = float(non_null.str.len().mean()) if len(non_null) else 0.0
        unique_vals = non_null.nunique()
        if avg_len > 60 and unique_vals > 40:
            excluded_text_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, bool_cols, categorical_cols, excluded_text_cols


def build_preprocessor(X_df: pd.DataFrame) -> Tuple[ColumnTransformer, Dict[str, Any]]:
    X_work = X_df.copy()
    numeric_cols, bool_cols, categorical_cols, excluded_text_cols = infer_feature_groups(X_work)

    for col in bool_cols:
        X_work[col] = X_work[col].astype("Int64")
    numeric_plus_bool = numeric_cols + bool_cols

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_plus_bool),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    info = {
        "numeric_columns": numeric_cols,
        "boolean_columns": bool_cols,
        "categorical_columns": categorical_cols,
        "excluded_text_columns": excluded_text_cols,
    }
    return preprocessor, info


def save_preprocessor(preprocessor: ColumnTransformer) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")


def _parser_rules_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    options_payload = format_response_options(
        response_options=row.get("response_options"),
        response_options_json=row.get("response_options_json"),
        response_type=row.get("response_type"),
        scale_guidance=row.get("scale_guidance"),
        help_text=row.get("help_text"),
        question=row.get("question_text_primary")
        or row.get("caregiver_question")
        or row.get("psychologist_question"),
        min_value=row.get("min_value"),
        max_value=row.get("max_value"),
    )
    return {
        "response_type": str(row.get("response_type", "")).lower(),
        "scale_type": options_payload["scale_type"],
        "range": [safe_float(row.get("min_value")), safe_float(row.get("max_value"))],
        "options": options_payload["options_list"],
        "human_options_text": options_payload["human_options_text"],
    }


def build_feature_schema(
    selected_features: List[str],
    feature_to_metadata: Dict[str, Dict[str, Any]],
    persist: bool = True,
) -> Dict[str, Any]:
    schema_items: List[Dict[str, Any]] = []
    for feature in selected_features:
        row = feature_to_metadata.get(feature, {})
        options_payload = format_response_options(
            response_options=row.get("response_options"),
            response_options_json=row.get("response_options_json"),
            response_type=row.get("response_type"),
            scale_guidance=row.get("scale_guidance"),
            help_text=row.get("help_text"),
            question=row.get("question_text_primary")
            or row.get("caregiver_question")
            or row.get("psychologist_question"),
            min_value=row.get("min_value"),
            max_value=row.get("max_value"),
        )
        response_options = options_payload["options_list"]

        question_text = (
            row.get("caregiver_question")
            or row.get("psychologist_question")
            or row.get("question_text_primary")
            or "¿Puedes describir esta situación con tus palabras?"
        )
        item = {
            "feature": feature,
            "questionnaire_item_id": row.get("questionnaire_item_id"),
            "question_text_primary": row.get("question_text_primary"),
            "caregiver_question": row.get("caregiver_question"),
            "psychologist_question": row.get("psychologist_question"),
            "feature_label_human": row.get("feature_label_human"),
            "feature_description": row.get("feature_description"),
            "term_explanation": row.get("term_explanation"),
            "section_name": row.get("section_name"),
            "subsection_name": row.get("subsection_name"),
            "domains_final": row.get("domains_final"),
            "feature_type": row.get("feature_type"),
            "feature_role": row.get("feature_role"),
            "response_type": row.get("response_type"),
            "response_options": response_options,
            "min_value": row.get("min_value"),
            "max_value": row.get("max_value"),
            "unit": row.get("unit"),
            "scale_guidance": row.get("scale_guidance"),
            "help_text": row.get("help_text"),
            "who_answers": row.get("who_answers"),
            "respondent_expected": row.get("respondent_expected"),
            "administered_by": row.get("administered_by"),
            "requires_clinician_administration": as_bool(
                row.get("requires_clinician_administration")
            ),
            "requires_child_self_report": as_bool(row.get("requires_child_self_report")),
            "question_audit_status": row.get("question_audit_status"),
            "is_required": as_bool(row.get("show_in_questionnaire_yes_no")),
            "parser_rules": _parser_rules_from_row({**row, "response_options": response_options}),
            "fallback_question": question_text,
            "human_options_text": options_payload["human_options_text"],
            "quick_chips": options_payload["quick_chips"],
            "scale_type": options_payload["scale_type"],
        }
        schema_items.append(item)

    schema = {
        "features": schema_items,
        "feature_order": selected_features,
        "fields": SCHEMA_FIELDS,
    }
    if persist:
        save_json(ARTIFACTS_DIR / "feature_schema.json", schema)
    return schema
