"""Human-friendly question explanations without technical leakage."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .response_options import format_response_options
from .utils import normalize_text


TECHNICAL_PATTERNS = [
    r"\binput for\b",
    r"\bfeature\b",
    r"\bresponse_options_json\b",
    r"\btarget_domain_\w+\b",
    r"\b[a-z]+_[a-z0-9_]+\b",  # snake_case
    r"\badhd\s+\w+\s+\d+\b",
]


def _looks_technical(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return True
    for pattern in TECHNICAL_PATTERNS:
        if re.search(pattern, norm):
            return True
    return False


def _safe_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return "" if _looks_technical(text) else text


def _build_simple_explanation(question: str, candidates: List[str]) -> str:
    for candidate in candidates:
        cleaned = _safe_text(candidate)
        if cleaned:
            return cleaned
    question_safe = _safe_text(question)
    if question_safe:
        return (
            "Esta pregunta busca entender una conducta observable en situaciones cotidianas, "
            "para ubicarla en una escala clara."
        )
    return "Esta pregunta busca entender mejor una situación observada en la vida diaria."


def _examples_by_scale(scale_type: str, question: str) -> List[str]:
    q = normalize_text(question)
    if any(token in q for token in ["pelea", "golpe", "amenaza", "intimida", "agrede"]):
        return [
            "Se considera cuando inicia agresiones, no cuando solo se defiende.",
            "Por ejemplo: empuja, golpea o provoca peleas de forma intencional.",
            "Si fue un hecho aislado de defensa, puedes aclararlo en tu respuesta.",
        ]

    if scale_type == "binary":
        return [
            "Marca Sí cuando la conducta se observa de forma repetida.",
            "Marca No cuando no ocurre o es excepcional.",
            "Si tienes duda, primero describe la situación y luego te ayudo a ubicarla.",
        ]
    if scale_type == "temporal_0_2":
        return [
            "No ocurrió: no se ha presentado.",
            "Ocurrió antes: pasó en el último año, pero no en los últimos 6 meses.",
            "Ocurrió recientemente: se presentó en los últimos 6 meses.",
        ]
    if scale_type == "frequency_0_3":
        return [
            "Nunca: no se observa.",
            "Ocasional: aparece pocas veces.",
            "Frecuente o casi siempre: se nota de forma repetida.",
        ]
    if scale_type == "impact_0_3":
        return [
            "Sin impacto: no afecta la vida diaria.",
            "Leve o moderado: afecta algo, pero sigue funcionando.",
            "Marcado: afecta mucho escuela, casa o convivencia.",
        ]
    if scale_type == "observation_0_2":
        return [
            "No se observa: no lo has visto.",
            "A veces: aparece en algunas situaciones o hay duda.",
            "Claramente: se observa de forma persistente.",
        ]
    return [
        "Piensa en situaciones recientes de casa, colegio o comunidad.",
        "Responde de forma breve con lo que has observado.",
        "Si lo necesitas, usa ejemplos concretos para describir la situación.",
    ]


def _expected_answer(scale_type: str, human_options_text: str) -> str:
    base = human_options_text.rstrip(".")
    if scale_type == "binary":
        return f"{base}. También puedes explicarlo con tus palabras."
    if scale_type in {"temporal_0_2", "frequency_0_3", "impact_0_3", "observation_0_2"}:
        return f"{base}. Si tienes duda, describe brevemente y te ayudo a ubicarla."
    return "Puedes responder con tus palabras de forma clara y breve."


def explain_question(feature_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    question = (
        metadata.get("caregiver_question")
        or metadata.get("psychologist_question")
        or metadata.get("question_text_primary")
        or metadata.get("fallback_question")
        or ""
    )
    question = str(question).strip()

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

    explanation_candidates = [
        metadata.get("help_text"),
        metadata.get("term_explanation"),
        metadata.get("feature_description"),
    ]
    simple_explanation = _build_simple_explanation(question, explanation_candidates)
    examples = _examples_by_scale(options_payload["scale_type"], question)

    return {
        "feature_name": feature_name,
        "simple_explanation": simple_explanation,
        "examples": examples,
        "not_about": [
            "No se trata de juzgar al niño.",
            "No confirma un diagnóstico clínico.",
            "No busca culpar a la familia.",
        ],
        "expected_answer": _expected_answer(
            options_payload["scale_type"],
            options_payload["human_options_text"],
        ),
        "human_options_text": options_payload["human_options_text"],
        "quick_chips": options_payload["quick_chips"],
        "scale_type": options_payload["scale_type"],
    }

