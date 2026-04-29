"""Human question explanation without exposing technical metadata."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .response_options import format_response_options
from .utils import normalize_text


TECH_PATTERNS = [
    r"\binput for\b",
    r"\bresponse_options_json\b",
    r"\bfeature_name\b",
    r"\btarget_domain_\w+\b",
    r"\b[a-z]+_[a-z0-9_]+\b",
    r"\badhd\s*\d*\b",
]


def _looks_technical(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    for pattern in TECH_PATTERNS:
        if re.search(pattern, norm):
            return True
    return False


def _safe_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if _looks_technical(text):
        return ""
    return text


def _fallback_explanation(question: str, section_name: str) -> str:
    q = normalize_text(question)
    section = normalize_text(section_name)

    if any(term in q for term in ["pelea", "agres", "golpe", "amenaz", "intimid"]):
        return (
            "Esta pregunta busca identificar conductas de conflicto que se inician activamente, "
            "diferenciandolas de situaciones en las que solo se responde para defenderse."
        )

    if any(term in q for term in ["escuela", "colegio", "norma", "convivencia"]):
        return (
            "Esta pregunta explora comportamientos observables en la convivencia diaria y si "
            "afectan relaciones, normas o rutina."
        )

    if "conduct" in section or "convivencia" in section:
        return (
            "Esta pregunta ayuda a entender comportamientos observables relacionados con convivencia "
            "y normas, para una estimacion preliminar."
        )

    return (
        "Esta pregunta busca ubicar una conducta observada en una escala clara para apoyar "
        "una estimacion orientativa."
    )


def _build_examples(scale_type: str, question: str) -> List[str]:
    q = normalize_text(question)

    if any(term in q for term in ["pelea", "agres", "golpe", "amenaz", "intimid"]):
        return [
            "Cuenta si el nino inicia el conflicto y no solo si responde para defenderse.",
            "Ejemplo: provoca, empuja o amenaza de forma intencional.",
            "Si fue algo aislado, puedes decirlo para dar contexto.",
        ]

    if scale_type == "binary":
        return [
            "Marca Si cuando la conducta se observa de forma repetida.",
            "Marca No cuando no se observa o fue algo excepcional.",
            "Si tienes duda, puedes describirlo brevemente.",
        ]

    if scale_type == "temporal_0_2":
        return [
            "No ocurrio: no se ha presentado.",
            "Ocurrio antes: paso en el ultimo ano, pero no en los ultimos 6 meses.",
            "Ocurrio recientemente: paso en los ultimos 6 meses.",
        ]

    if scale_type == "frequency_0_3":
        return [
            "Nunca: no se observa.",
            "Ocasional: aparece algunas veces.",
            "Frecuente o casi siempre: se repite en muchas situaciones.",
        ]

    if scale_type == "observation_0_2":
        return [
            "No se observa: no lo has visto.",
            "A veces: aparece en algunas situaciones o hay duda.",
            "Claramente: se observa de forma evidente y persistente.",
        ]

    if scale_type == "impact_0_3":
        return [
            "Sin impacto: no altera la rutina diaria.",
            "Leve o moderado: afecta en parte algunas actividades.",
            "Marcado: interfiere mucho en casa, escuela o convivencia.",
        ]

    return [
        "Piensa en situaciones recientes de casa, escuela o convivencia.",
        "Responde con observaciones concretas, no con etiquetas.",
        "Si quieres, da un ejemplo corto para contextualizar.",
    ]


def _expected_answer(scale_type: str, human_options_text: str) -> str:
    if scale_type in {"binary", "temporal_0_2", "frequency_0_3", "observation_0_2", "impact_0_3"}:
        return f"{human_options_text} Tambien puedes responder con tus palabras."
    if scale_type == "numeric_range":
        return "Responde con un numero. Si quieres, puedes agregar una breve aclaracion."
    return "Responde de forma breve y clara con tus palabras."


def explain_question(feature_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    question = str(
        metadata.get("question")
        or metadata.get("caregiver_question")
        or metadata.get("psychologist_question")
        or metadata.get("question_text_primary")
        or metadata.get("fallback_question")
        or "¿Puedes contarme que has observado en esta situacion?"
    ).strip()

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

    candidates = [
        _safe_text(metadata.get("help_text")),
        _safe_text(metadata.get("term_explanation")),
        _safe_text(metadata.get("feature_description")),
    ]
    simple_explanation = next((item for item in candidates if item), "")
    if not simple_explanation:
        simple_explanation = _fallback_explanation(question, str(metadata.get("section_name") or ""))

    return {
        "feature_name": feature_name,
        "simple_explanation": simple_explanation,
        "examples": _build_examples(options_payload["scale_type"], question),
        "not_about": [
            "No se trata de juzgar al nino.",
            "No significa un diagnostico confirmado.",
            "No busca culpar a la familia.",
        ],
        "expected_answer": _expected_answer(options_payload["scale_type"], options_payload["human_options_text"]),
        "human_options_text": options_payload["human_options_text"],
        "quick_chips": options_payload["quick_chips"],
        "scale_type": options_payload["scale_type"],
    }
