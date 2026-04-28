"""Central configuration and path helpers for the local PGC project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
LOGS_DIR = ROOT_DIR / "logs"
TESTS_DIR = ROOT_DIR / "tests"
DOCS_DIR = ROOT_DIR / "docs"
WEB_DIR = ROOT_DIR / "web"
WEB_TEMPLATES_DIR = WEB_DIR / "templates"
WEB_STATIC_DIR = WEB_DIR / "static"
WEB_STATIC_CSS_DIR = WEB_STATIC_DIR / "css"
WEB_STATIC_JS_DIR = WEB_STATIC_DIR / "js"
WEB_STATIC_IMG_DIR = WEB_STATIC_DIR / "img"

REQUIRED_DIRS: List[Path] = [
    DATA_DIR,
    SRC_DIR,
    MODELS_DIR,
    ARTIFACTS_DIR,
    LOGS_DIR,
    TESTS_DIR,
    DOCS_DIR,
    WEB_DIR,
    WEB_TEMPLATES_DIR,
    WEB_STATIC_DIR,
    WEB_STATIC_CSS_DIR,
    WEB_STATIC_JS_DIR,
    WEB_STATIC_IMG_DIR,
]

MAIN_DATASET_PREFERRED_NAMES: List[str] = [
    "hybrid_no_external_scores_dataset_ready(1).csv",
    "hybrid_no_external_scores_dataset_ready.csv",
]

QUESTIONNAIRE_PREFERRED_NAMES: List[str] = [
    "questionnaire_v16_4_visible_questions_excel_utf8.csv",
]

TARGET_COLUMNS: List[str] = [
    "target_domain_adhd_final",
    "target_domain_conduct_final",
    "target_domain_elimination_final",
    "target_domain_anxiety_final",
    "target_domain_depression_final",
]

DEFAULT_TARGET_COLUMN = "target_domain_conduct_final"

TARGET_DISORDER_MAP: Dict[str, str] = {
    "adhd": "target_domain_adhd_final",
    "conduct": "target_domain_conduct_final",
    "conducta": "target_domain_conduct_final",
    "trastorno de conducta": "target_domain_conduct_final",
    "elimination": "target_domain_elimination_final",
    "eliminacion": "target_domain_elimination_final",
    "eliminación": "target_domain_elimination_final",
    "anxiety": "target_domain_anxiety_final",
    "ansiedad": "target_domain_anxiety_final",
    "depression": "target_domain_depression_final",
    "depresion": "target_domain_depression_final",
    "depresión": "target_domain_depression_final",
}

CONDUCT_SECTION_NAME = "3. Comportamiento, normas y convivencia"

MEDICAL_DISCLAIMER = (
    "Este resultado no es un diagnóstico médico y debe ser revisado "
    "por un profesional calificado."
)


def ensure_required_dirs() -> None:
    """Create the expected project directories when missing."""
    for path in REQUIRED_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def resolve_target_column(
    available_targets: List[str] | None = None,
) -> Tuple[str, str]:
    """
    Resolve TARGET_COLUMN based on environment variable and available target columns.

    Returns:
        (target_column, selection_method)
    """
    available = set(available_targets or TARGET_COLUMNS)
    env_value = os.getenv("TARGET_DISORDER", "").strip().lower()
    if env_value:
        mapped = TARGET_DISORDER_MAP.get(env_value)
        if mapped and mapped in available:
            return mapped, f"TARGET_DISORDER={env_value}"
    if DEFAULT_TARGET_COLUMN in available:
        return DEFAULT_TARGET_COLUMN, "default_conduct"
    for target in TARGET_COLUMNS:
        if target in available:
            return target, "fallback_first_available_target"
    raise ValueError(
        "No se pudo seleccionar una columna objetivo. "
        "Verifica que exista alguna columna target_domain_* en el dataset."
    )
