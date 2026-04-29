"""Feature selection rules aligned with target domain and questionnaire contract."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import ARTIFACTS_DIR, CONDUCT_SECTION_NAME, DEFAULT_TARGET_COLUMN
from .utils import normalize_text, save_json


DEFAULT_CONDUCT_FEATURES = [
    "conduct_impairment_global",
    "conduct_onset_before_10",
    "conduct_lpe_01_lack_remorse_guilt",
    "conduct_lpe_02_callous_lack_empathy",
    "conduct_lpe_03_unconcerned_performance",
    "conduct_lpe_04_shallow_deficient_affect",
    "conduct_01_bullies_threatens_intimidates",
    "conduct_02_initiates_fights",
    "conduct_03_weapon_use",
    "conduct_04_physical_cruelty_people",
    "conduct_05_physical_cruelty_animals",
    "conduct_06_steals_confronting_victim",
    "conduct_07_forced_sex",
    "conduct_08_fire_setting",
    "conduct_09_property_destruction",
    "conduct_10_breaks_into_house_building_car",
    "conduct_11_lies_to_obtain_or_avoid",
    "conduct_12_steals_without_confrontation",
    "conduct_13_stays_out_at_night_before_13",
    "conduct_14_runs_away_overnight",
    "conduct_15_truancy_before_13",
    "age_years",
    "sex_assigned_at_birth",
]

SENSITIVE_ID_HINTS = [
    "participant_id",
    " id",
    "_id",
    "nombre",
    "documento",
    "correo",
    "telefono",
    "direccion",
]

TARGET_RULES: Dict[str, Dict[str, Any]] = {
    "target_domain_conduct_final": {
        "include_prefixes": ["conduct_"],
        "exclude_prefixes": [
            "adhd_",
            "anxiety_",
            "depression_",
            "elimination_",
            "agor_",
            "panic_",
            "gad_",
            "mdd_",
            "pdd_",
            "dmdd_",
            "ocd_",
        ],
        "allow_features": {"age_years", "sex_assigned_at_birth"},
        "section_keywords": ["comportamiento", "convivencia", "normas", "conducta"],
        "domain_keywords": ["conduct"],
    },
    "target_domain_adhd_final": {
        "include_prefixes": ["adhd_"],
        "exclude_prefixes": ["conduct_", "anxiety_", "depression_", "elimination_"],
        "allow_features": {"age_years", "sex_assigned_at_birth"},
        "section_keywords": ["atencion", "actividad", "impulsividad"],
        "domain_keywords": ["adhd"],
    },
    "target_domain_anxiety_final": {
        "include_prefixes": ["anxiety_", "agor_", "panic_", "gad_"],
        "exclude_prefixes": ["adhd_", "conduct_", "depression_", "elimination_"],
        "allow_features": {"age_years", "sex_assigned_at_birth"},
        "section_keywords": ["miedos", "preocupacion", "ansiedad"],
        "domain_keywords": ["anxiety"],
    },
    "target_domain_depression_final": {
        "include_prefixes": ["depression_", "mdd_", "pdd_", "dmdd_"],
        "exclude_prefixes": ["adhd_", "conduct_", "anxiety_", "elimination_"],
        "allow_features": {"age_years", "sex_assigned_at_birth"},
        "section_keywords": ["animo", "tristeza", "irritabilidad", "depres"],
        "domain_keywords": ["depression"],
    },
    "target_domain_elimination_final": {
        "include_prefixes": ["elimination_", "encop_", "enur_"],
        "exclude_prefixes": ["adhd_", "conduct_", "anxiety_", "depression_"],
        "allow_features": {"age_years", "sex_assigned_at_birth"},
        "section_keywords": ["eliminacion", "esfinter", "control"],
        "domain_keywords": ["elimination"],
    },
}


def _get_rules(target_col: str) -> Dict[str, Any]:
    return TARGET_RULES.get(target_col, TARGET_RULES[DEFAULT_TARGET_COLUMN])


def _has_prefix(feature: str, prefixes: List[str]) -> bool:
    return any(feature.startswith(prefix) for prefix in prefixes)


def _contains_keyword(text: str, keywords: List[str]) -> bool:
    norm = normalize_text(text)
    return any(keyword in norm for keyword in keywords)


def _should_exclude_col(col: str, target_col: str) -> Tuple[bool, str]:
    c = normalize_text(col)
    if col == target_col:
        return True, "target_actual"
    if col.startswith("target_domain_"):
        return True, "target_domain_exclusion"
    if c.startswith("eng_"):
        return True, "derived_eng_feature_exclusion"
    if c.endswith("_count"):
        return True, "derived_count_exclusion"
    if c.endswith("_sum") or c.endswith("_index") or c.endswith("_total"):
        return True, "derived_aggregate_exclusion"
    if "derived" in c or "composite" in c:
        return True, "derived_feature_exclusion"
    if any(h in c for h in SENSITIVE_ID_HINTS):
        return True, "identifier_exclusion"
    return False, ""


def is_feature_allowed_for_target(feature: str, meta: Dict[str, Any], target_col: str) -> Tuple[bool, str]:
    rules = _get_rules(target_col)
    feature_norm = normalize_text(feature)

    if feature in rules["allow_features"]:
        return True, "allowed_demographic"
    if _has_prefix(feature_norm, rules["exclude_prefixes"]):
        return False, "target_domain_prefix_exclusion"
    if _has_prefix(feature_norm, rules["include_prefixes"]):
        return True, "target_prefix_match"

    section_name = str(meta.get("section_name") or "")
    domains_final = str(meta.get("domains_final") or "")
    section_ok = _contains_keyword(section_name, rules["section_keywords"])
    domain_ok = _contains_keyword(domains_final, rules["domain_keywords"])

    if target_col == DEFAULT_TARGET_COLUMN:
        section_match_exact = normalize_text(section_name) == normalize_text(CONDUCT_SECTION_NAME)
        if section_match_exact and not _has_prefix(feature_norm, rules["exclude_prefixes"]):
            return True, "conduct_section_match"

    if domain_ok and section_ok:
        return True, "domain_and_section_match"
    if domain_ok and not _has_prefix(feature_norm, rules["exclude_prefixes"]):
        return True, "domain_match"
    return False, "out_of_target_domain"


def select_features(
    dataset_df: pd.DataFrame,
    questionnaire_df: pd.DataFrame,
    target_col: str,
) -> Dict[str, Any]:
    dataset_cols = set(dataset_df.columns)
    selected: List[str] = []
    excluded: List[Dict[str, str]] = []
    warnings: List[str] = []

    q_by_feature = {
        str(row["feature"]): row.to_dict()
        for _, row in questionnaire_df.iterrows()
        if str(row.get("feature", "")).strip()
    }

    # Priority 1: explicit conduct core when default target.
    if target_col == DEFAULT_TARGET_COLUMN:
        for feat in DEFAULT_CONDUCT_FEATURES:
            if feat in dataset_cols and feat in q_by_feature:
                excluded_flag, reason = _should_exclude_col(feat, target_col)
                if excluded_flag:
                    excluded.append({"feature": feat, "reason": reason})
                    continue
                allowed, allow_reason = is_feature_allowed_for_target(feat, q_by_feature[feat], target_col)
                if allowed:
                    selected.append(feat)
                else:
                    excluded.append({"feature": feat, "reason": allow_reason})

    # Priority 2: all questionnaire features consistent with target rules.
    for feat, row in q_by_feature.items():
        if feat not in dataset_cols:
            excluded.append({"feature": feat, "reason": "not_in_dataset"})
            continue
        should_exclude, reason = _should_exclude_col(feat, target_col)
        if should_exclude:
            excluded.append({"feature": feat, "reason": reason})
            continue
        allowed, allow_reason = is_feature_allowed_for_target(feat, row, target_col)
        if allowed:
            selected.append(feat)
        else:
            excluded.append({"feature": feat, "reason": allow_reason})

    selected = sorted(set(selected))
    if not selected:
        raise ValueError(
            "No se encontraron features útiles para entrenamiento según el contrato del cuestionario "
            "y las reglas del dominio objetivo."
        )

    rules = _get_rules(target_col)
    off_domain_selected = [
        feat for feat in selected if _has_prefix(feat, rules["exclude_prefixes"])
    ]
    if off_domain_selected:
        warnings.append(
            "Se detectaron features potencialmente fuera del dominio objetivo: "
            + ", ".join(off_domain_selected[:20])
        )

    origins: Dict[str, Dict[str, Any]] = {}
    for feat in selected:
        meta = q_by_feature.get(feat, {})
        origins[feat] = {
            "origin": "questionnaire_contract",
            "question": meta.get("caregiver_question")
            or meta.get("psychologist_question")
            or meta.get("question_text_primary")
            or "Pregunta no disponible",
            "domains_final": meta.get("domains_final"),
            "section_name": meta.get("section_name"),
        }

    report = {
        "target_used": target_col,
        "features_used": selected,
        "features_used_count": len(selected),
        "features_excluded": excluded,
        "feature_origins": origins,
        "warnings": warnings,
        "decision": (
            "Selección por dominio objetivo con prioridad de contrato, exclusión de fugas, "
            "y bloqueo de prefijos de otros dominios para evitar incoherencias."
        ),
    }
    save_json(ARTIFACTS_DIR / "selected_features.json", report)
    return report
