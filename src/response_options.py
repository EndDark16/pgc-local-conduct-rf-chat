"""Human-friendly response options normalization and scale inference."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional

from .utils import normalize_text, safe_float, try_parse_json


def _as_python_object(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (list, dict)):
        return raw
    text = str(raw).strip()
    if not text or normalize_text(text) in {"nan", "none", "null"}:
        return None
    parsed = try_parse_json(text)
    if parsed is not None:
        return parsed
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _to_int_if_integer_like(value: Any) -> Any:
    number = safe_float(value)
    if number is None:
        return value
    if float(number).is_integer():
        return int(number)
    return float(number)


def _normalize_options_list(raw_options: Any) -> List[Dict[str, Any]]:
    obj = _as_python_object(raw_options)
    options: List[Dict[str, Any]] = []

    if obj is None:
        return options

    if isinstance(obj, dict):
        if "value" in obj or "label" in obj:
            value = _to_int_if_integer_like(obj.get("value"))
            label = str(obj.get("label") or obj.get("text") or value or "").strip()
            if label:
                options.append({"value": value, "label": label})
            return options
        for key, val in obj.items():
            label = str(val).strip()
            value = _to_int_if_integer_like(key)
            if label:
                options.append({"value": value, "label": label})
        return options

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                value = _to_int_if_integer_like(item.get("value"))
                label = str(
                    item.get("label")
                    or item.get("text")
                    or item.get("name")
                    or item.get("option")
                    or item.get("description")
                    or value
                    or ""
                ).strip()
                if label:
                    options.append({"value": value, "label": label})
            else:
                value = _to_int_if_integer_like(item)
                label = str(item).strip()
                if label:
                    options.append({"value": value, "label": label})
        return options

    # Fallback: parse informal string patterns.
    text = str(obj).strip()
    if not text:
        return options
    chunks = [chunk.strip() for chunk in re.split(r"[;|,/]", text) if chunk.strip()]
    for chunk in chunks:
        match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[:=\-]\s*(.+)$", chunk)
        if match:
            value = _to_int_if_integer_like(match.group(1))
            label = match.group(2).strip()
            options.append({"value": value, "label": label})
        else:
            options.append({"value": None, "label": chunk})
    return options


def _contains_any(text: str, words: List[str]) -> bool:
    return any(word in text for word in words)


def infer_scale_type(
    response_type: Any = None,
    response_options: Any = None,
    scale_guidance: Any = None,
    help_text: Any = None,
    question: Any = None,
    min_value: Any = None,
    max_value: Any = None,
) -> str:
    response_type_norm = normalize_text(response_type)
    options_list = _normalize_options_list(response_options)
    values = [
        int(opt["value"])
        for opt in options_list
        if isinstance(opt.get("value"), (int, float)) and float(opt["value"]).is_integer()
    ]
    value_set = set(values)
    labels_text = " ".join(normalize_text(opt.get("label")) for opt in options_list)
    context_text = " ".join(
        [
            response_type_norm,
            normalize_text(scale_guidance),
            normalize_text(help_text),
            normalize_text(question),
            labels_text,
        ]
    )

    if _contains_any(response_type_norm, ["yes_no", "boolean", "binary", "si_no"]):
        return "binary"
    if value_set == {0, 1} and _contains_any(
        context_text, ["si", "sí", "no", "afirmativo", "negativo", "verdadero", "falso"]
    ):
        return "binary"

    temporal_words = [
        "ultimo 6",
        "ultimos 6",
        "6 meses",
        "12 meses",
        "ocurrio",
        "ocurri",
        "reciente",
        "no recientemente",
    ]
    observation_words = ["se observa", "claramente", "hay duda", "duda", "persistente"]
    impact_words = ["impacto", "moderado", "marcado", "sin impacto", "afecta"]
    frequency_words = ["nunca", "ocasional", "frecuente", "casi siempre", "muy marcado"]

    if value_set == {0, 1, 2} and _contains_any(context_text, temporal_words):
        return "temporal_0_2"
    if value_set == {0, 1, 2} and _contains_any(context_text, observation_words):
        return "observation_0_2"
    if value_set == {0, 1, 2, 3}:
        if _contains_any(response_type_norm, ["frequency", "frecuencia"]):
            return "frequency_0_3"
        if _contains_any(response_type_norm, ["impact"]):
            return "impact_0_3"
        has_frequency = _contains_any(context_text, frequency_words)
        has_impact = _contains_any(context_text, impact_words)
        if has_frequency and not has_impact:
            return "frequency_0_3"
        if has_impact and not has_frequency:
            return "impact_0_3"
        if has_frequency and has_impact:
            # Prefer frequency unless context is explicitly about functional impact.
            if _contains_any(context_text, ["afecta", "impacto", "interfiere"]):
                return "impact_0_3"
            return "frequency_0_3"

    min_num = safe_float(min_value)
    max_num = safe_float(max_value)
    if min_num is not None and max_num is not None:
        if _contains_any(response_type_norm, ["numeric", "number", "int", "float", "range"]):
            return "numeric_range"

    if options_list:
        return "categorical"
    return "unknown"


def _labels_to_sentence(labels: List[str]) -> str:
    clean = [str(label).strip() for label in labels if str(label).strip()]
    if not clean:
        return "Puedes responder con tus palabras."
    if len(clean) == 1:
        return f"Puedes responder: {clean[0]}."
    if len(clean) == 2:
        return f"Puedes responder: {clean[0]} o {clean[1]}."
    return f"Puedes responder: {', '.join(clean[:-1])}, o {clean[-1]}."


def _canonical_templates(scale_type: str) -> Dict[str, Any]:
    templates = {
        "binary": {
            "human_options_text": "Puedes responder: No o Sí.",
            "quick_chips": ["Sí", "No", "No entiendo"],
        },
        "frequency_0_3": {
            "human_options_text": (
                "Puedes responder: nunca, leve u ocasional, claro o frecuente, "
                "o muy marcado/casi siempre."
            ),
            "quick_chips": ["Nunca", "Ocasional", "Frecuente", "Casi siempre", "No entiendo"],
        },
        "temporal_0_2": {
            "human_options_text": "Puedes responder: no ocurrió, ocurrió antes, u ocurrió recientemente.",
            "quick_chips": ["No ocurrió", "Ocurrió antes", "Ocurrió recientemente", "No entiendo"],
        },
        "observation_0_2": {
            "human_options_text": "Puedes responder: no se observa, a veces, o claramente.",
            "quick_chips": ["No se observa", "A veces", "Claramente", "No entiendo"],
        },
        "impact_0_3": {
            "human_options_text": "Puedes responder: no afecta, leve, moderado, o marcado.",
            "quick_chips": ["No afecta", "Leve", "Moderado", "Marcado", "No entiendo"],
        },
    }
    return templates.get(
        scale_type,
        {
            "human_options_text": "Puedes responder con tus palabras.",
            "quick_chips": ["No entiendo"],
        },
    )


def format_response_options(
    response_options: Any = None,
    response_options_json: Any = None,
    response_type: Any = None,
    scale_guidance: Any = None,
    help_text: Any = None,
    question: Any = None,
    min_value: Any = None,
    max_value: Any = None,
) -> Dict[str, Any]:
    options_raw = response_options if response_options is not None else response_options_json
    options_list = _normalize_options_list(options_raw)

    scale_type = infer_scale_type(
        response_type=response_type,
        response_options=options_list,
        scale_guidance=scale_guidance,
        help_text=help_text,
        question=question,
        min_value=min_value,
        max_value=max_value,
    )

    template = _canonical_templates(scale_type)
    if options_list and scale_type == "categorical":
        labels = [str(opt.get("label") or opt.get("value")) for opt in options_list]
        human_text = _labels_to_sentence(labels)
        quick = labels[:4] + ["No entiendo"]
    elif options_list and scale_type == "binary":
        human_text = "Puedes responder: No o Sí."
        quick = ["Sí", "No", "No entiendo"]
    else:
        human_text = template["human_options_text"]
        quick = template["quick_chips"]

    # Deduplicate chips preserving order.
    quick_chips: List[str] = []
    seen = set()
    for chip in quick:
        chip_text = str(chip).strip()
        if not chip_text:
            continue
        key = normalize_text(chip_text)
        if key in seen:
            continue
        seen.add(key)
        quick_chips.append(chip_text)

    return {
        "options_list": options_list,
        "human_options_text": human_text,
        "quick_chips": quick_chips,
        "scale_type": scale_type,
    }
