"""Local NLP interpreter with scale-aware rules and human clarifications."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from .response_options import format_response_options, infer_scale_type as infer_scale_type_from_options
from .utils import normalize_text as _normalize_text, safe_float


HELP_PHRASES = [
    "no entiendo",
    "que significa",
    "explicame",
    "no se que responder",
    "no se como responder",
    "puedes explicarlo",
    "dame un ejemplo",
]

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
    text = _normalize_text(value)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_user_does_not_understand(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    if any(phrase in norm for phrase in HELP_PHRASES):
        return True
    fuzzy_target = ["no entiendo", "explicame", "que significa", "no se"]
    best = process.extractOne(norm, fuzzy_target, scorer=fuzz.partial_ratio)
    return bool(best and best[1] >= 88)


def is_help_request(text: str) -> bool:
    return detect_user_does_not_understand(text)


def _has_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _extract_numeric_value(text: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return float(match.group(0))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return float(value)
    return None


def infer_scale_type(metadata: Dict[str, Any]) -> str:
    return infer_scale_type_from_options(
        response_type=metadata.get("response_type"),
        response_options=metadata.get("response_options") or metadata.get("response_options_json"),
        scale_guidance=metadata.get("scale_guidance"),
        help_text=metadata.get("help_text"),
        question=metadata.get("question_text_primary")
        or metadata.get("caregiver_question")
        or metadata.get("psychologist_question"),
        min_value=metadata.get("min_value"),
        max_value=metadata.get("max_value"),
    )


def parse_binary_answer(text: str) -> Dict[str, Any]:
    positives = [
        "si",
        "si ocurre",
        "si suele pasar",
        "claro",
        "correcto",
        "afirmativo",
        "verdadero",
        "se observa",
        "pasa",
        "suele pasar",
        "frecuentemente si",
        "casi siempre si",
    ]
    negatives = [
        "no",
        "no ocurre",
        "nunca",
        "falso",
        "negativo",
        "no se observa",
        "no suele pasar",
        "casi nunca",
        "para nada",
    ]
    ambiguous = [
        "a veces",
        "algunas veces",
        "depende",
        "no se",
        "tal vez",
        "puede ser",
        "en ocasiones",
        "no estoy seguro",
    ]

    if _has_any(text, ambiguous):
        return {
            "parsed_value": None,
            "confidence": 0.35,
            "needs_clarification": True,
            "clarification_question": (
                "Entiendo que ocurre algunas veces. Para esta pregunta necesito una respuesta más directa: "
                "¿dirías que sí se observa de forma repetida en la vida diaria, o que no?"
            ),
            "reasoning_summary": "La respuesta es intermedia para una escala binaria.",
            "user_friendly_interpretation": "Todavía no estoy seguro porque la respuesta quedó entre sí y no.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
        }

    if _has_any(text, positives):
        return {
            "parsed_value": 1,
            "confidence": 0.93,
            "needs_clarification": False,
            "clarification_question": "",
            "reasoning_summary": "Se detectó una respuesta afirmativa.",
            "user_friendly_interpretation": "Lo entendí como: sí se observa.",
            "value_explanation": "Para esta escala equivale al valor 1.",
            "answer_category": "respuesta clara",
        }
    if _has_any(text, negatives):
        return {
            "parsed_value": 0,
            "confidence": 0.93,
            "needs_clarification": False,
            "clarification_question": "",
            "reasoning_summary": "Se detectó una respuesta negativa.",
            "user_friendly_interpretation": "Lo entendí como: no se observa.",
            "value_explanation": "Para esta escala equivale al valor 0.",
            "answer_category": "respuesta clara",
        }
    return {
        "parsed_value": None,
        "confidence": 0.33,
        "needs_clarification": True,
        "clarification_question": (
            "Entiendo que tu respuesta no es completamente sí o no. Para esta pregunta necesito saber "
            "si se observa de forma repetida: ¿sí o no?"
        ),
        "reasoning_summary": "No se encontró una afirmación o negación directa.",
        "user_friendly_interpretation": "Todavía no estoy seguro de si la respuesta corresponde a sí o no.",
        "value_explanation": "",
        "answer_category": "respuesta ambigua",
    }


def parse_frequency_0_3(text: str) -> Dict[str, Any]:
    temporal_words = [
        "ocurrio recientemente",
        "ocurrio hace poco",
        "en los ultimos 6 meses",
        "actualmente",
        "hace poco",
    ]
    if _has_any(text, temporal_words):
        return {
            "parsed_value": None,
            "confidence": 0.32,
            "needs_clarification": True,
            "clarification_question": (
                "Eso me dice cuándo ocurrió, pero esta pregunta necesita saber qué tan frecuente o marcado es. "
                "¿Dirías que nunca, leve/ocasional, frecuente o casi siempre?"
            ),
            "reasoning_summary": "Se detectó temporalidad en una escala de frecuencia.",
            "user_friendly_interpretation": "Todavía no lo puedo ubicar en frecuencia.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
        }

    level_terms = {
        0: ["nunca", "no ocurre", "ausente", "para nada", "casi nunca"],
        1: ["leve", "ocasional", "rara vez", "pocas veces", "muy poco", "de vez en cuando", "a veces"],
        2: ["frecuente", "frecuentemente", "claro", "varias veces", "muchas veces", "a menudo", "se nota", "bastante"],
        3: [
            "muy marcado",
            "casi siempre",
            "siempre",
            "todo el tiempo",
            "constantemente",
            "muy frecuente",
            "demasiado",
            "muchisimo",
            "casi todo el tiempo",
        ],
    }
    for level, terms in level_terms.items():
        if _has_any(text, terms):
            if level == 1 and "a veces" in text:
                return {
                    "parsed_value": 1,
                    "confidence": 0.69,
                    "needs_clarification": True,
                    "clarification_question": (
                        "Entendí que ocurre de forma ocasional. Lo guardaré como nivel 1, "
                        "a menos que quieras corregirlo."
                    ),
                    "reasoning_summary": "La expresión 'a veces' se aproxima a nivel ocasional.",
                    "user_friendly_interpretation": "Lo entendí como: ocasional.",
                    "value_explanation": "Para esta escala equivale al nivel 1.",
                    "answer_category": "respuesta parcialmente clara",
                }
            confidence = 0.94 if level in {0, 3} else 0.88
            return {
                "parsed_value": level,
                "confidence": confidence,
                "needs_clarification": False,
                "clarification_question": "",
                "reasoning_summary": (
                    "Se detectó una expresión equivalente a 'casi siempre', que corresponde al nivel más alto de frecuencia."
                    if level == 3 and ("casi siempre" in text or "todo el tiempo" in text)
                    else "Se detectó una expresión de frecuencia compatible con la escala."
                ),
                "user_friendly_interpretation": (
                    "Lo entendí como: casi siempre." if level == 3 else f"Lo entendí como nivel {level} de frecuencia."
                ),
                "value_explanation": f"Para esta escala equivale al nivel {level}.",
                "answer_category": "respuesta clara",
            }

    return {
        "parsed_value": None,
        "confidence": 0.3,
        "needs_clarification": True,
        "clarification_question": (
            "Necesito ubicarlo en una escala de frecuencia. ¿Se parece más a nunca, "
            "leve/ocasional, frecuente o casi siempre?"
        ),
        "reasoning_summary": "No se detectó una expresión clara de frecuencia.",
        "user_friendly_interpretation": "Todavía no estoy seguro de cómo ubicar esta respuesta en la escala.",
        "value_explanation": "",
        "answer_category": "respuesta ambigua",
    }


def parse_temporal_0_2(text: str) -> Dict[str, Any]:
    level_terms = {
        0: ["no ocurrio", "nunca", "no ha pasado", "jamas", "no se ha presentado"],
        1: [
            "ocurrio antes",
            "antes pasaba",
            "hace tiempo",
            "hace mas de seis meses",
            "hace mas de 6 meses",
            "el ano pasado",
            "paso pero ya no",
            "no recientemente",
        ],
        2: [
            "ocurrio recientemente",
            "actualmente",
            "pasa ahora",
            "en los ultimos seis meses",
            "en los ultimos 6 meses",
            "este semestre",
            "hace poco",
            "recientemente",
            "sigue pasando",
        ],
    }
    for level, terms in level_terms.items():
        if _has_any(text, terms):
            return {
                "parsed_value": level,
                "confidence": 0.92,
                "needs_clarification": False,
                "clarification_question": "",
                "reasoning_summary": "Se identificó una referencia temporal compatible con la escala.",
                "user_friendly_interpretation": (
                    "Lo entendí como: ocurrió recientemente."
                    if level == 2
                    else "Lo entendí como: ocurrió antes."
                    if level == 1
                    else "Lo entendí como: no ocurrió."
                ),
                "value_explanation": f"Para esta escala equivale al valor {level}.",
                "answer_category": "respuesta clara",
            }
    return {
        "parsed_value": None,
        "confidence": 0.34,
        "needs_clarification": True,
        "clarification_question": (
            "Necesito saber cuándo ocurrió. ¿No ocurrió, ocurrió antes pero no recientemente, "
            "u ocurrió en los últimos 6 meses?"
        ),
        "reasoning_summary": "No se pudo ubicar temporalmente la respuesta.",
        "user_friendly_interpretation": "Todavía no sé en qué momento ubicar esta respuesta.",
        "value_explanation": "",
        "answer_category": "respuesta ambigua",
    }


def parse_observation_0_2(text: str) -> Dict[str, Any]:
    level_terms = {
        0: ["no se observa", "no", "nunca", "no pasa", "no lo he visto"],
        1: ["a veces", "algunas veces", "hay duda", "no estoy seguro", "ocasionalmente", "puede ser"],
        2: ["claramente", "si se observa", "frecuente", "persistente", "casi siempre", "muy claro", "sin duda"],
    }
    for level, terms in level_terms.items():
        if _has_any(text, terms):
            confidence = 0.9 if level in {0, 2} else 0.76
            return {
                "parsed_value": level,
                "confidence": confidence,
                "needs_clarification": confidence < 0.8,
                "clarification_question": (
                    "Entendí que se observa a veces. ¿Quieres que lo guarde como nivel 1?"
                    if confidence < 0.8
                    else ""
                ),
                "reasoning_summary": "Se detectó una expresión de observación compatible con la escala.",
                "user_friendly_interpretation": (
                    "Lo entendí como: claramente se observa."
                    if level == 2
                    else "Lo entendí como: se observa a veces."
                    if level == 1
                    else "Lo entendí como: no se observa."
                ),
                "value_explanation": f"Para esta escala equivale al valor {level}.",
                "answer_category": "respuesta clara" if confidence >= 0.8 else "respuesta parcialmente clara",
            }
    return {
        "parsed_value": None,
        "confidence": 0.3,
        "needs_clarification": True,
        "clarification_question": "Necesito ubicarlo como: no se observa, a veces, o claramente.",
        "reasoning_summary": "No se identificó un nivel claro de observación.",
        "user_friendly_interpretation": "Todavía no pude ubicar esta respuesta en la escala de observación.",
        "value_explanation": "",
        "answer_category": "respuesta ambigua",
    }


def parse_impact_0_3(text: str) -> Dict[str, Any]:
    level_terms = {
        0: ["sin impacto", "no afecta", "nada", "no hay problema"],
        1: ["leve", "poco", "afecta un poco", "manejable"],
        2: ["moderado", "medio", "afecta bastante", "interfiere"],
        3: ["marcado", "fuerte", "grave", "severo", "afecta mucho", "impide actividades"],
    }
    for level, terms in level_terms.items():
        if _has_any(text, terms):
            return {
                "parsed_value": level,
                "confidence": 0.9,
                "needs_clarification": False,
                "clarification_question": "",
                "reasoning_summary": "Se detectó un nivel de impacto compatible con la escala.",
                "user_friendly_interpretation": f"Lo entendí como impacto nivel {level}.",
                "value_explanation": f"Para esta escala equivale al valor {level}.",
                "answer_category": "respuesta clara",
            }
    return {
        "parsed_value": None,
        "confidence": 0.3,
        "needs_clarification": True,
        "clarification_question": (
            "Necesito saber cuánto afecta. ¿No afecta, afecta poco, afecta moderadamente o afecta mucho?"
        ),
        "reasoning_summary": "No se identificó un nivel claro de impacto.",
        "user_friendly_interpretation": "Todavía no estoy seguro de cuánto impacto describes.",
        "value_explanation": "",
        "answer_category": "respuesta ambigua",
    }


def parse_numeric_range(text: str, min_value: Any, max_value: Any) -> Dict[str, Any]:
    parsed = _extract_numeric_value(text)
    min_num = safe_float(min_value)
    max_num = safe_float(max_value)
    if parsed is None:
        return {
            "parsed_value": None,
            "confidence": 0.32,
            "needs_clarification": True,
            "clarification_question": "Para esta pregunta necesito un número.",
            "reasoning_summary": "No se detectó ningún valor numérico.",
            "user_friendly_interpretation": "Todavía no encontré un número en tu respuesta.",
            "value_explanation": "",
            "answer_category": "respuesta insuficiente",
        }
    if min_num is not None and parsed < min_num:
        return {
            "parsed_value": None,
            "confidence": 0.28,
            "needs_clarification": True,
            "clarification_question": f"El valor debe ser igual o mayor a {min_num}. ¿Puedes ajustarlo?",
            "reasoning_summary": "El valor está por debajo del rango permitido.",
            "user_friendly_interpretation": "El número parece menor a lo permitido.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
        }
    if max_num is not None and parsed > max_num:
        return {
            "parsed_value": None,
            "confidence": 0.28,
            "needs_clarification": True,
            "clarification_question": f"El valor debe ser igual o menor a {max_num}. ¿Puedes ajustarlo?",
            "reasoning_summary": "El valor está por encima del rango permitido.",
            "user_friendly_interpretation": "El número parece mayor a lo permitido.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
        }
    value = int(parsed) if float(parsed).is_integer() else float(parsed)
    return {
        "parsed_value": value,
        "confidence": 0.9,
        "needs_clarification": False,
        "clarification_question": "",
        "reasoning_summary": "Se detectó un número válido dentro del rango esperado.",
        "user_friendly_interpretation": f"Lo entendí como: {value}.",
        "value_explanation": "Se guardará ese valor numérico.",
        "answer_category": "respuesta clara",
    }


def parse_categorical(text: str, options_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not options_list:
        return {
            "parsed_value": None,
            "confidence": 0.25,
            "needs_clarification": True,
            "clarification_question": "Necesito una respuesta más concreta para esta pregunta.",
            "reasoning_summary": "No hay opciones categóricas definidas.",
            "user_friendly_interpretation": "No tengo opciones suficientes para interpretar esta respuesta.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
        }

    labels = [normalize_text(opt.get("label")) for opt in options_list]
    best = process.extractOne(text, labels, scorer=fuzz.ratio)
    if not best or best[1] < 72:
        return {
            "parsed_value": None,
            "confidence": 0.34,
            "needs_clarification": True,
            "clarification_question": "Necesito una respuesta más cercana a las opciones de esta pregunta.",
            "reasoning_summary": "No hubo coincidencia suficiente con las opciones.",
            "user_friendly_interpretation": "Todavía no encontré una opción clara.",
            "value_explanation": "",
            "answer_category": "respuesta ambigua",
        }
    _, score, index = best
    option = options_list[index]
    value = option.get("value", option.get("label"))
    label = str(option.get("label") or value)
    return {
        "parsed_value": value,
        "confidence": round(min(0.93, score / 100.0), 4),
        "needs_clarification": score < 85,
        "clarification_question": (
            f"Entendí la opción '{label}'. ¿Es correcto?" if score < 85 else ""
        ),
        "reasoning_summary": "Se encontró coincidencia con una opción categórica.",
        "user_friendly_interpretation": f"Lo entendí como: {label}.",
        "value_explanation": "Guardaré esa opción para el modelo.",
        "answer_category": "respuesta clara" if score >= 85 else "respuesta parcialmente clara",
    }


def build_clarification_question(scale_type: str) -> str:
    mapping = {
        "binary": "Para esta pregunta necesito un sí o no claro.",
        "frequency_0_3": "Necesito ubicarlo en frecuencia: nunca, ocasional, frecuente o casi siempre.",
        "temporal_0_2": "Necesito saber cuándo ocurrió: no ocurrió, ocurrió antes, u ocurrió recientemente.",
        "impact_0_3": "Necesito saber cuánto afecta: no afecta, leve, moderado o marcado.",
        "observation_0_2": "Necesito ubicarlo como: no se observa, a veces o claramente.",
        "numeric_range": "Para esta pregunta necesito un número dentro del rango esperado.",
    }
    return mapping.get(
        scale_type,
        "Necesito una respuesta un poco más clara para continuar.",
    )


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
        question=metadata.get("question_text_primary")
        or metadata.get("caregiver_question")
        or metadata.get("psychologist_question"),
        min_value=metadata.get("min_value"),
        max_value=metadata.get("max_value"),
    )
    scale_type = infer_scale_type(metadata)

    base = {
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
        base.update(
            {
                "confidence": 0.2,
                "clarification_question": build_clarification_question(scale_type),
                "reasoning_summary": "La respuesta está vacía.",
                "user_friendly_interpretation": "Necesito una respuesta para poder continuar.",
                "answer_category": "respuesta insuficiente",
            }
        )
        return base

    if detect_user_does_not_understand(norm):
        base.update(
            {
                "confidence": 0.99,
                "clarification_question": "Te explico la pregunta con más detalle y ejemplos.",
                "reasoning_summary": "El usuario pidió ayuda para entender la pregunta.",
                "user_friendly_interpretation": "Entiendo que necesitas una explicación antes de responder.",
                "answer_category": "usuario no entendió la pregunta",
            }
        )
        return base

    if scale_type == "binary":
        parsed = parse_binary_answer(norm)
    elif scale_type == "frequency_0_3":
        parsed = parse_frequency_0_3(norm)
    elif scale_type == "temporal_0_2":
        parsed = parse_temporal_0_2(norm)
    elif scale_type == "observation_0_2":
        parsed = parse_observation_0_2(norm)
    elif scale_type == "impact_0_3":
        parsed = parse_impact_0_3(norm)
    elif scale_type == "numeric_range":
        parsed = parse_numeric_range(norm, metadata.get("min_value"), metadata.get("max_value"))
    else:
        parsed = parse_categorical(norm, options_payload["options_list"])

    base.update(parsed)
    base["confidence"] = round(float(base.get("confidence", 0.0)), 4)

    if base["parsed_value"] is not None:
        min_num = safe_float(metadata.get("min_value"))
        max_num = safe_float(metadata.get("max_value"))
        try:
            parsed_float = float(base["parsed_value"])
            if min_num is not None and parsed_float < min_num:
                base["validation_error"] = f"El valor está por debajo del mínimo permitido ({min_num})."
            if max_num is not None and parsed_float > max_num:
                base["validation_error"] = f"El valor está por encima del máximo permitido ({max_num})."
        except (TypeError, ValueError):
            pass

    if base["validation_error"]:
        base["parsed_value"] = None
        base["needs_clarification"] = True
        base["confidence"] = min(base["confidence"], 0.35)
        base["clarification_question"] = build_clarification_question(scale_type)
        base["answer_category"] = "respuesta ambigua"
        base["user_friendly_interpretation"] = "Necesito ajustar la respuesta al rango permitido."

    if base["needs_clarification"] and attempt >= max_attempts:
        base["clarification_question"] = (
            "No he podido interpretar esta respuesta con suficiente seguridad. "
            "Por favor responde de forma más directa usando una de estas opciones: "
            "no ocurrió, ocurrió antes pero no recientemente, ocurrió en los últimos 6 meses, sí o no, según corresponda."
        )

    return base

