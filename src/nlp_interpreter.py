"""Generalized local NLP interpreter for conversational questionnaire answers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from .response_options import format_response_options, infer_scale_type as infer_scale_type_from_metadata
from .utils import normalize_text as base_normalize_text, safe_float


HELP_TERMS = [
    "no entiendo",
    "no comprendo",
    "explicame",
    "explicar",
    "que significa",
    "que quiere decir",
    "dame un ejemplo",
    "ejemplo",
    "ayuda",
    "no se que responder",
    "explicame con palabras simples",
]

AFFIRMATION_TERMS = [
    "si",
    "si ocurre",
    "si se observa",
    "si afecta",
    "si suele pasar",
    "claro",
    "correcto",
    "afirmativo",
    "verdadero",
    "se observa",
    "suele pasar",
    "ha pasado",
    "frecuentemente si",
    "casi siempre si",
]

NEGATION_TERMS = [
    "no",
    "no ocurre",
    "nunca",
    "falso",
    "negativo",
    "no se observa",
    "no suele pasar",
    "casi nunca",
    "para nada",
    "no afecta",
    "no ha pasado",
]

UNCERTAINTY_TERMS = [
    "a veces",
    "depende",
    "no se",
    "no estoy seguro",
    "no estoy segura",
    "tal vez",
    "puede ser",
    "en ocasiones",
    "algunas veces",
    "quizas",
]

TEMPORAL_NONE_TERMS = [
    "no ocurrio",
    "nunca",
    "jamas",
    "no ha pasado",
    "no se ha presentado",
]
TEMPORAL_PAST_TERMS = [
    "ocurrio antes",
    "antes pasaba",
    "hace tiempo",
    "hace mas de seis meses",
    "hace mas de 6 meses",
    "el ano pasado",
    "paso pero ya no",
    "no recientemente",
]
TEMPORAL_RECENT_TERMS = [
    "ocurrio recientemente",
    "recientemente",
    "actualmente",
    "pasa ahora",
    "en los ultimos 6 meses",
    "ultimos seis meses",
    "hace poco",
    "sigue pasando",
]

FREQUENCY_TERMS = {
    0: ["nunca", "no ocurre", "ausente", "para nada", "casi nunca"],
    1: ["leve", "ocasional", "rara vez", "pocas veces", "muy poco", "de vez en cuando", "a veces"],
    2: ["frecuente", "frecuentemente", "claro", "varias veces", "muchas veces", "a menudo", "se nota", "bastante"],
    3: ["muy marcado", "casi siempre", "siempre", "todo el tiempo", "constantemente", "muy frecuente", "muchisimo", "casi todo el tiempo"],
}

OBSERVATION_TERMS = {
    0: ["no se observa", "no", "nunca", "no pasa", "no lo he visto"],
    1: ["a veces", "algunas veces", "hay duda", "no estoy seguro", "ocasionalmente", "puede ser"],
    2: ["claramente", "si se observa", "frecuente", "persistente", "casi siempre", "muy claro", "sin duda"],
}

IMPACT_TERMS = {
    0: ["sin impacto", "no afecta", "nada", "no hay problema"],
    1: ["leve", "poco", "afecta un poco", "manejable"],
    2: ["moderado", "medio", "afecta bastante", "interfiere"],
    3: ["marcado", "fuerte", "grave", "severo", "afecta mucho", "impide actividades"],
}

OCCURRENCE_TERMS = ["ocurrio", "paso", "ha pasado", "se presento", "sucede", "aparece"]

NUMBER_WORDS = {
    "cero": 0,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
}


def normalize_text(value: Any) -> str:
    text = base_normalize_text(value)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    typo_map = {
        "a veses": "a veces",
        "ocurio": "ocurrio",
        "reciente mente": "recientemente",
        "noce": "no se",
    }
    for wrong, fixed in typo_map.items():
        text = text.replace(wrong, fixed)
    return text


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _any_phrase(text: str, phrases: List[str]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def _best_fuzzy(text: str, terms: List[str]) -> Tuple[str, float]:
    if not text or not terms:
        return "", 0.0
    best = process.extractOne(text, terms, scorer=fuzz.partial_ratio)
    if not best:
        return "", 0.0
    return str(best[0]), float(best[1])


def detect_help_intent(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    if _any_phrase(norm, HELP_TERMS):
        return True
    _, score = _best_fuzzy(norm, HELP_TERMS)
    return score >= 91.0


def detect_user_does_not_understand(text: str) -> bool:
    return detect_help_intent(text)


def detect_affirmation(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    if _any_phrase(norm, AFFIRMATION_TERMS):
        return True
    fuzzy_terms = [term for term in AFFIRMATION_TERMS if len(term.replace(" ", "")) >= 4]
    _, score = _best_fuzzy(norm, fuzzy_terms)
    return score >= 88.0


def detect_negation(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    if _any_phrase(norm, NEGATION_TERMS):
        return True
    negation_hint = re.search(r"\b(no|nunca|jamas|negativo|falso|nada)\b", norm) is not None
    if not negation_hint:
        return False
    fuzzy_terms = [term for term in NEGATION_TERMS if len(term.replace(" ", "")) >= 4]
    _, score = _best_fuzzy(norm, fuzzy_terms)
    return score >= 88.0


def detect_uncertainty(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    if _any_phrase(norm, UNCERTAINTY_TERMS):
        return True
    _, score = _best_fuzzy(norm, UNCERTAINTY_TERMS)
    return score >= 86.0


def detect_occurrence(text: str) -> Optional[bool]:
    norm = normalize_text(text)
    if detect_negation(norm):
        return False
    if _any_phrase(norm, OCCURRENCE_TERMS) or detect_affirmation(norm):
        return True
    if detect_uncertainty(norm):
        return None
    return None


def detect_temporal_reference(text: str) -> Optional[str]:
    norm = normalize_text(text)
    if _any_phrase(norm, TEMPORAL_NONE_TERMS):
        return "none"
    if _any_phrase(norm, TEMPORAL_PAST_TERMS):
        return "past"
    if _any_phrase(norm, TEMPORAL_RECENT_TERMS):
        return "recent"
    return None


def _detect_level(text: str, map_terms: Dict[int, List[str]], fuzzy_threshold: float) -> Optional[int]:
    norm = normalize_text(text)
    for level, terms in map_terms.items():
        if _any_phrase(norm, terms):
            return level

    best_level: Optional[int] = None
    best_score = 0.0
    for level, terms in map_terms.items():
        _, score = _best_fuzzy(norm, terms)
        if score > best_score:
            best_score = score
            best_level = level
    if best_level is not None and best_score >= fuzzy_threshold:
        return best_level
    return None


def detect_frequency_level(text: str) -> Optional[int]:
    return _detect_level(text, FREQUENCY_TERMS, fuzzy_threshold=82.0)


def detect_observation_level(text: str) -> Optional[int]:
    return _detect_level(text, OBSERVATION_TERMS, fuzzy_threshold=82.0)


def detect_impact_level(text: str) -> Optional[int]:
    return _detect_level(text, IMPACT_TERMS, fuzzy_threshold=82.0)


def detect_binary_value(text: str) -> Optional[int]:
    norm = normalize_text(text)
    aff = detect_affirmation(norm)
    neg = detect_negation(norm)

    if aff and not neg:
        return 1
    if neg and not aff:
        return 0
    return None


def _extract_number(text: str) -> Optional[float]:
    norm = normalize_text(text)
    match = re.search(r"-?\d+(?:\.\d+)?", norm)
    if match:
        return float(match.group(0))

    for word, value in NUMBER_WORDS.items():
        if _contains_phrase(norm, word):
            return float(value)
    return None


def infer_scale_type(metadata: Dict[str, Any]) -> str:
    return infer_scale_type_from_metadata(
        response_type=metadata.get("response_type"),
        response_options=metadata.get("response_options") or metadata.get("response_options_json"),
        scale_guidance=metadata.get("scale_guidance"),
        help_text=metadata.get("help_text"),
        question=metadata.get("question")
        or metadata.get("question_text_primary")
        or metadata.get("caregiver_question")
        or metadata.get("psychologist_question"),
        min_value=metadata.get("min_value"),
        max_value=metadata.get("max_value"),
    )


def build_contextual_clarification(scale_type: str, context: Optional[Dict[str, Any]] = None) -> str:
    context = context or {}
    if scale_type == "binary":
        if context.get("uncertain"):
            return (
                "Entiendo que ocurre algunas veces. Para esta pregunta necesito una respuesta mas directa: "
                "¿dirias que si se observa de forma repetida en la vida diaria, o que no?"
            )
        return "Para esta pregunta necesito confirmar si la respuesta es si o no."

    if scale_type == "temporal_0_2":
        if context.get("occurrence") is True:
            return (
                "Entiendo que si ocurrio. Para ubicarlo bien, necesito saber si fue algo anterior "
                "o si ocurrio recientemente."
            )
        return (
            "Necesito saber cuando ocurrio: no ocurrio, ocurrio antes pero no recientemente, "
            "u ocurrio en los ultimos 6 meses."
        )

    if scale_type == "frequency_0_3":
        if context.get("temporal_only"):
            return (
                "Eso me dice cuando ocurrio, pero esta pregunta necesita saber que tan frecuente o marcado es. "
                "¿Dirias que nunca, leve u ocasional, frecuente, o casi siempre?"
            )
        return "Necesito ubicarlo en frecuencia: nunca, ocasional, frecuente o casi siempre."

    if scale_type == "observation_0_2":
        return "Necesito ubicarlo como: no se observa, a veces, o claramente."

    if scale_type == "impact_0_3":
        return "Necesito saber cuanto afecta: no afecta, leve, moderado o marcado."

    if scale_type == "numeric_range":
        return "Para esta pregunta necesito un numero dentro del rango esperado."

    if scale_type == "categorical":
        return "Necesito una respuesta mas cercana a las opciones de esta pregunta."

    return "Necesito una respuesta un poco mas clara para continuar."


def parse_binary_answer(text: str) -> Dict[str, Any]:
    parsed = detect_binary_value(text)
    if parsed == 1:
        return {
            "parsed_value": 1,
            "confidence": 0.92,
            "needs_clarification": False,
            "reasoning_summary": "Se detecto una afirmacion clara para una escala binaria.",
            "user_friendly_interpretation": "Lo entendi como: si.",
            "value_explanation": "Para esta escala equivale al valor 1.",
            "answer_category": "respuesta clara",
            "context": {},
        }
    if parsed == 0:
        return {
            "parsed_value": 0,
            "confidence": 0.92,
            "needs_clarification": False,
            "reasoning_summary": "Se detecto una negacion clara para una escala binaria.",
            "user_friendly_interpretation": "Lo entendi como: no.",
            "value_explanation": "Para esta escala equivale al valor 0.",
            "answer_category": "respuesta clara",
            "context": {},
        }

    if detect_uncertainty(text):
        return {
            "parsed_value": None,
            "confidence": 0.45,
            "needs_clarification": True,
            "reasoning_summary": "La respuesta expresa duda para una pregunta binaria.",
            "user_friendly_interpretation": "Entiendo que ocurre en algunas situaciones, pero necesito confirmarlo.",
            "value_explanation": "",
            "answer_category": "respuesta parcialmente clara",
            "context": {"uncertain": True},
        }

    return {
        "parsed_value": None,
        "confidence": 0.35,
        "needs_clarification": True,
        "reasoning_summary": "No se detecto una respuesta binaria clara.",
        "user_friendly_interpretation": "Todavia no estoy seguro de si corresponde a si o no.",
        "value_explanation": "",
        "answer_category": "respuesta ambigua",
        "context": {},
    }


def parse_temporal_0_2(text: str) -> Dict[str, Any]:
    reference = detect_temporal_reference(text)
    occurrence = detect_occurrence(text)

    if reference == "none" or occurrence is False:
        return {
            "parsed_value": 0,
            "confidence": 0.9,
            "needs_clarification": False,
            "reasoning_summary": "Se detecto que no ocurrio.",
            "user_friendly_interpretation": "Lo entendi como: no ocurrio.",
            "value_explanation": "Para esta escala equivale al valor 0.",
            "answer_category": "respuesta clara",
            "context": {},
        }
    if reference == "past":
        return {
            "parsed_value": 1,
            "confidence": 0.9,
            "needs_clarification": False,
            "reasoning_summary": "Se detecto ocurrencia anterior no reciente.",
            "user_friendly_interpretation": "Lo entendi como: ocurrio antes, pero no recientemente.",
            "value_explanation": "Para esta escala equivale al valor 1.",
            "answer_category": "respuesta clara",
            "context": {},
        }
    if reference == "recent":
        return {
            "parsed_value": 2,
            "confidence": 0.92,
            "needs_clarification": False,
            "reasoning_summary": "Se detecto ocurrencia reciente.",
            "user_friendly_interpretation": "Lo entendi como: ocurrio recientemente.",
            "value_explanation": "Para esta escala equivale al valor 2.",
            "answer_category": "respuesta clara",
            "context": {},
        }

    if occurrence is True:
        return {
            "parsed_value": None,
            "confidence": 0.62,
            "needs_clarification": True,
            "reasoning_summary": "Se detecto ocurrencia, pero falta temporalidad.",
            "user_friendly_interpretation": "Entiendo que si ocurrio, pero aun necesito cuando.",
            "value_explanation": "",
            "answer_category": "respuesta parcialmente clara",
            "context": {"occurrence": True},
        }

    return {
        "parsed_value": None,
        "confidence": 0.35,
        "needs_clarification": True,
        "reasoning_summary": "No se logro identificar la temporalidad.",
        "user_friendly_interpretation": "Todavia no puedo ubicar cuando ocurrio.",
        "value_explanation": "",
        "answer_category": "respuesta ambigua",
        "context": {},
    }


def parse_frequency_0_3(text: str) -> Dict[str, Any]:
    reference = detect_temporal_reference(text)
    if reference in {"past", "recent"}:
        return {
            "parsed_value": None,
            "confidence": 0.35,
            "needs_clarification": True,
            "reasoning_summary": "La respuesta describe temporalidad y no frecuencia.",
            "user_friendly_interpretation": "Entiendo cuando ocurrio, pero aqui necesito que tan frecuente es.",
            "value_explanation": "",
            "answer_category": "respuesta parcialmente clara",
            "context": {"temporal_only": True},
        }

    level = detect_frequency_level(text)
    if level is None:
        if detect_uncertainty(text):
            return {
                "parsed_value": 1,
                "confidence": 0.67,
                "needs_clarification": True,
                "reasoning_summary": "Se detecto una frecuencia ocasional con duda.",
                "user_friendly_interpretation": "Lo entendi como: ocasional.",
                "value_explanation": "Para esta escala equivale al nivel 1, salvo que quieras corregirlo.",
                "answer_category": "respuesta parcialmente clara",
                "context": {},
            }
        return {
            "parsed_value": None,
            "confidence": 0.34,
            "needs_clarification": True,
            "reasoning_summary": "No se detecto frecuencia clara.",
            "user_friendly_interpretation": "Todavia no puedo ubicar esta respuesta en frecuencia.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
            "context": {},
        }

    confidence = {0: 0.88, 1: 0.72, 2: 0.84, 3: 0.92}.get(level, 0.8)
    return {
        "parsed_value": level,
        "confidence": confidence,
        "needs_clarification": confidence < 0.65,
        "reasoning_summary": "Se detecto una expresion compatible con escala de frecuencia.",
        "user_friendly_interpretation": (
            "Lo entendi como: casi siempre." if level == 3 else "Lo entendi como: frecuente." if level == 2 else "Lo entendi como: ocasional." if level == 1 else "Lo entendi como: nunca."
        ),
        "value_explanation": f"Para esta escala equivale al nivel {level}.",
        "answer_category": "respuesta clara" if confidence >= 0.8 else "respuesta parcialmente clara",
        "context": {},
    }


def parse_observation_0_2(text: str) -> Dict[str, Any]:
    level = detect_observation_level(text)
    if level is None:
        return {
            "parsed_value": None,
            "confidence": 0.33,
            "needs_clarification": True,
            "reasoning_summary": "No se detecto nivel de observacion claro.",
            "user_friendly_interpretation": "Todavia no puedo ubicar esta respuesta en observacion.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
            "context": {},
        }

    confidence = {0: 0.9, 1: 0.72, 2: 0.9}.get(level, 0.8)
    return {
        "parsed_value": level,
        "confidence": confidence,
        "needs_clarification": confidence < 0.65,
        "reasoning_summary": "Se detecto una expresion compatible con escala de observacion.",
        "user_friendly_interpretation": "Lo entendi como: no se observa." if level == 0 else "Lo entendi como: se observa a veces." if level == 1 else "Lo entendi como: se observa claramente.",
        "value_explanation": f"Para esta escala equivale al valor {level}.",
        "answer_category": "respuesta clara" if confidence >= 0.8 else "respuesta parcialmente clara",
        "context": {},
    }


def parse_impact_0_3(text: str) -> Dict[str, Any]:
    level = detect_impact_level(text)
    if level is None:
        return {
            "parsed_value": None,
            "confidence": 0.33,
            "needs_clarification": True,
            "reasoning_summary": "No se detecto nivel de impacto claro.",
            "user_friendly_interpretation": "Todavia no puedo ubicar cuanto impacto describe la respuesta.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
            "context": {},
        }

    confidence = {0: 0.9, 1: 0.82, 2: 0.84, 3: 0.9}.get(level, 0.82)
    return {
        "parsed_value": level,
        "confidence": confidence,
        "needs_clarification": confidence < 0.65,
        "reasoning_summary": "Se detecto una expresion compatible con escala de impacto.",
        "user_friendly_interpretation": "Lo entendi como: sin impacto." if level == 0 else "Lo entendi como: impacto leve." if level == 1 else "Lo entendi como: impacto moderado." if level == 2 else "Lo entendi como: impacto marcado.",
        "value_explanation": f"Para esta escala equivale al nivel {level}.",
        "answer_category": "respuesta clara" if confidence >= 0.8 else "respuesta parcialmente clara",
        "context": {},
    }


def parse_numeric_range(text: str, min_value: Any, max_value: Any) -> Dict[str, Any]:
    value = _extract_number(text)
    min_num = safe_float(min_value)
    max_num = safe_float(max_value)

    if value is None:
        return {
            "parsed_value": None,
            "confidence": 0.3,
            "needs_clarification": True,
            "reasoning_summary": "No se detecto un numero en la respuesta.",
            "user_friendly_interpretation": "Necesito un numero para registrar esta respuesta.",
            "value_explanation": "",
            "answer_category": "respuesta insuficiente",
            "context": {},
        }

    if min_num is not None and value < min_num:
        return {
            "parsed_value": None,
            "confidence": 0.3,
            "needs_clarification": True,
            "reasoning_summary": "El valor esta por debajo del rango permitido.",
            "user_friendly_interpretation": "El numero parece menor al minimo permitido.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
            "validation_error": f"El valor esta por debajo del minimo permitido ({min_num}).",
            "context": {},
        }

    if max_num is not None and value > max_num:
        return {
            "parsed_value": None,
            "confidence": 0.3,
            "needs_clarification": True,
            "reasoning_summary": "El valor esta por encima del rango permitido.",
            "user_friendly_interpretation": "El numero parece mayor al maximo permitido.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
            "validation_error": f"El valor esta por encima del maximo permitido ({max_num}).",
            "context": {},
        }

    parsed = int(value) if float(value).is_integer() else float(value)
    return {
        "parsed_value": parsed,
        "confidence": 0.92,
        "needs_clarification": False,
        "reasoning_summary": "Se detecto un valor numerico valido.",
        "user_friendly_interpretation": f"Lo entendi como: {parsed}.",
        "value_explanation": "Guardare ese valor numerico.",
        "answer_category": "respuesta clara",
        "context": {},
    }


def parse_categorical(text: str, options_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not options_list:
        return {
            "parsed_value": None,
            "confidence": 0.3,
            "needs_clarification": True,
            "reasoning_summary": "No hay opciones disponibles para mapear.",
            "user_friendly_interpretation": "Necesito una respuesta mas concreta para esta pregunta.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
            "context": {},
        }

    labels_norm = [normalize_text(opt.get("label")) for opt in options_list]
    best = process.extractOne(normalize_text(text), labels_norm, scorer=fuzz.ratio)
    if not best:
        return {
            "parsed_value": None,
            "confidence": 0.3,
            "needs_clarification": True,
            "reasoning_summary": "No hubo coincidencia con las opciones disponibles.",
            "user_friendly_interpretation": "No encontre una opcion clara en tu respuesta.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
            "context": {},
        }

    _label, score, idx = best
    if float(score) < 72.0:
        return {
            "parsed_value": None,
            "confidence": round(float(score) / 100.0, 3),
            "needs_clarification": True,
            "reasoning_summary": "La coincidencia con opciones fue baja.",
            "user_friendly_interpretation": "Necesito una respuesta mas cercana a las opciones sugeridas.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
            "context": {},
        }

    opt = options_list[int(idx)]
    value = opt.get("value") if opt.get("value") is not None else opt.get("label")
    label = str(opt.get("label") or value)
    conf = min(0.92, float(score) / 100.0)
    return {
        "parsed_value": value,
        "confidence": round(conf, 3),
        "needs_clarification": conf < 0.65,
        "reasoning_summary": "Se detecto una coincidencia con una opcion valida.",
        "user_friendly_interpretation": f"Lo entendi como: {label}.",
        "value_explanation": "Guardare esa opcion para el modelo.",
        "answer_category": "respuesta clara" if conf >= 0.8 else "respuesta parcialmente clara",
        "context": {},
    }


def interpret_open_answer(scale_type: str, text: str, metadata: Dict[str, Any], options_payload: Dict[str, Any]) -> Dict[str, Any]:
    if scale_type == "binary":
        return parse_binary_answer(text)
    if scale_type == "temporal_0_2":
        return parse_temporal_0_2(text)
    if scale_type == "frequency_0_3":
        return parse_frequency_0_3(text)
    if scale_type == "observation_0_2":
        return parse_observation_0_2(text)
    if scale_type == "impact_0_3":
        return parse_impact_0_3(text)
    if scale_type == "numeric_range":
        return parse_numeric_range(text, metadata.get("min_value"), metadata.get("max_value"))
    return parse_categorical(text, options_payload.get("options_list", []))


def interpret_answer(
    feature_name: str,
    raw_answer: str,
    metadata: Dict[str, Any],
    attempt: int = 1,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    raw = str(raw_answer or "").strip()
    norm = normalize_text(raw)

    options_payload = format_response_options(
        response_options=metadata.get("response_options"),
        response_options_json=metadata.get("response_options_json"),
        response_type=metadata.get("response_type"),
        scale_guidance=metadata.get("scale_guidance"),
        help_text=metadata.get("help_text"),
        question=metadata.get("question")
        or metadata.get("question_text_primary")
        or metadata.get("caregiver_question")
        or metadata.get("psychologist_question"),
        min_value=metadata.get("min_value"),
        max_value=metadata.get("max_value"),
    )
    scale_type = infer_scale_type(metadata)

    result = {
        "feature_name": feature_name,
        "raw_answer": raw,
        "parsed_value": None,
        "expected_type": scale_type,
        "confidence": 0.0,
        "needs_clarification": True,
        "clarification_question": "",
        "reasoning_summary": "",
        "user_friendly_interpretation": "",
        "value_explanation": "",
        "validation_error": "",
        "answer_category": "respuesta ambigua",
        "human_options_text": options_payload["human_options_text"],
        "quick_chips": options_payload["quick_chips"],
        "scale_type": scale_type,
    }

    if not norm:
        result.update(
            {
                "confidence": 0.2,
                "needs_clarification": True,
                "clarification_question": build_contextual_clarification(scale_type),
                "reasoning_summary": "La respuesta esta vacia.",
                "user_friendly_interpretation": "Necesito una respuesta para continuar.",
                "answer_category": "respuesta insuficiente",
            }
        )
        return result

    if detect_help_intent(norm):
        result.update(
            {
                "confidence": 0.99,
                "needs_clarification": True,
                "clarification_question": "Te explico esta pregunta en palabras mas simples.",
                "reasoning_summary": "Se detecto una solicitud de ayuda explicita.",
                "user_friendly_interpretation": "Entiendo que quieres una explicacion antes de responder.",
                "answer_category": "usuario no entendio la pregunta",
            }
        )
        return result

    parsed = interpret_open_answer(scale_type, norm, metadata, options_payload)
    context = parsed.pop("context", {})
    result.update(parsed)
    result["confidence"] = round(float(result.get("confidence", 0.0)), 4)

    # Additional confidence rule.
    if result.get("parsed_value") is not None and result["confidence"] < 0.65:
        result["needs_clarification"] = True
        if result["answer_category"] == "respuesta clara":
            result["answer_category"] = "respuesta parcialmente clara"

    if result.get("parsed_value") is not None:
        min_num = safe_float(metadata.get("min_value"))
        max_num = safe_float(metadata.get("max_value"))
        try:
            value_float = float(result["parsed_value"])
            if min_num is not None and value_float < min_num:
                result["validation_error"] = f"El valor esta por debajo del minimo permitido ({min_num})."
            if max_num is not None and value_float > max_num:
                result["validation_error"] = f"El valor esta por encima del maximo permitido ({max_num})."
        except (TypeError, ValueError):
            pass

    if result.get("validation_error"):
        result["parsed_value"] = None
        result["needs_clarification"] = True
        result["answer_category"] = "respuesta ambigua"

    if result.get("needs_clarification"):
        result["clarification_question"] = build_contextual_clarification(scale_type, context)

    if result.get("needs_clarification") and attempt >= max_attempts:
        result["clarification_question"] = (
            "No he podido interpretar esta respuesta con suficiente seguridad. "
            "Por favor responde de forma directa usando una de las opciones sugeridas."
        )

    return result


def is_help_request(text: str) -> bool:
    return detect_help_intent(text)
