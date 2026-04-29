"""Humanized questionnaire generation for chat UI."""

from __future__ import annotations

from typing import Any, Dict, List

from .response_options import format_response_options
from .utils import normalize_text


def _auto_fallback_question() -> str:
    return "En tu experiencia reciente, ¿qué has observado sobre esta situación?"


def _pick_question_text(metadata: Dict[str, Any], role: str) -> str:
    role_norm = normalize_text(role)
    if role_norm in {"psychologist", "psicologo", "psicólogo"}:
        preferred = metadata.get("psychologist_question")
    else:
        preferred = metadata.get("caregiver_question")
    return str(preferred or metadata.get("question_text_primary") or metadata.get("fallback_question") or _auto_fallback_question()).strip()


def question_for_feature(metadata: Dict[str, Any], role: str = "caregiver") -> Dict[str, Any]:
    question = _pick_question_text(metadata, role=role)
    options_payload = format_response_options(
        response_options=metadata.get("response_options"),
        response_options_json=metadata.get("response_options_json"),
        response_type=metadata.get("response_type"),
        scale_guidance=metadata.get("scale_guidance"),
        help_text=metadata.get("help_text"),
        question=question,
        min_value=metadata.get("min_value"),
        max_value=metadata.get("max_value"),
    )

    return {
        "feature": str(metadata.get("feature", "")).strip(),
        "question": question,
        "feature_label_human": metadata.get("feature_label_human"),
        "help_text": metadata.get("help_text") or "",
        "scale_guidance": metadata.get("scale_guidance") or "",
        "response_options": options_payload["options_list"],
        "human_options_text": options_payload["human_options_text"],
        "quick_chips": options_payload["quick_chips"],
        "scale_type": options_payload["scale_type"],
        "section_name": metadata.get("section_name"),
        "subsection_name": metadata.get("subsection_name"),
    }


def generate_questionnaire(schema: Dict[str, Any], role: str = "caregiver") -> List[Dict[str, Any]]:
    items = schema.get("features", [])
    return [question_for_feature(item, role=role) for item in items]
