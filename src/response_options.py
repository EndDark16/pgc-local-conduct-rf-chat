"""Human formatting for response options and robust scale inference."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, Iterable, List, Sequence

from .utils import normalize_text, safe_float, try_parse_json


def _load_raw_object(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list, tuple)):
        return raw

    text = str(raw).strip()
    if not text:
        return None

    parsed = try_parse_json(text)
    if parsed is not None:
        return parsed

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass

    # irregular formats: "0=No;1=Si" or "No|Si"
    return text


def _number_or_original(value: Any) -> Any:
    num = safe_float(value)
    if num is None:
        return value
    if float(num).is_integer():
        return int(num)
    return float(num)


def _clean_label(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _from_dict(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    if "options" in obj and isinstance(obj["options"], (list, tuple)):
        return _normalize_option_list(obj["options"])

    if "choices" in obj and isinstance(obj["choices"], (list, tuple)):
        return _normalize_option_list(obj["choices"])

    if "value" in obj or "label" in obj:
        label = _clean_label(obj.get("label") or obj.get("text") or obj.get("name") or obj.get("value"))
        if label:
            out.append({"value": _number_or_original(obj.get("value")), "label": label})
        return out

    for key, value in obj.items():
        if isinstance(value, dict):
            nested = _normalize_option_list(value)
            out.extend(nested)
            continue
        label = _clean_label(value)
        if not label:
            continue
        out.append({"value": _number_or_original(key), "label": label})
    return out


def _from_string(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    chunks = [chunk.strip() for chunk in re.split(r"[\n;|]", text) if chunk.strip()]

    for chunk in chunks:
        # 0: No / 1=Si / 2 - Reciente
        pair = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(?::|=|-)\s*(.+)$", chunk)
        if pair:
            out.append({"value": _number_or_original(pair.group(1)), "label": _clean_label(pair.group(2))})
            continue

        # fallback plain option
        label = _clean_label(chunk)
        if label:
            out.append({"value": None, "label": label})

    if not out:
        # comma fallback
        for item in [i.strip() for i in text.split(",") if i.strip()]:
            out.append({"value": None, "label": _clean_label(item)})
    return out


def _normalize_option_list(raw: Any) -> List[Dict[str, Any]]:
    obj = _load_raw_object(raw)
    if obj is None:
        return []

    if isinstance(obj, dict):
        options = _from_dict(obj)
    elif isinstance(obj, (list, tuple)):
        options = []
        for item in obj:
            if isinstance(item, dict):
                options.extend(_from_dict(item))
            else:
                label = _clean_label(item)
                if label:
                    options.append({"value": _number_or_original(item), "label": label})
    else:
        options = _from_string(str(obj))

    # Deduplicate by normalized label/value.
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for option in options:
        label = _clean_label(option.get("label"))
        if not label:
            continue
        value = option.get("value")
        key = (normalize_text(label), str(value))
        if key in seen:
            continue
        seen.add(key)
        dedup.append({"value": value, "label": label})
    return dedup


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _numeric_set(options: Iterable[Dict[str, Any]]) -> set[int]:
    values: set[int] = set()
    for option in options:
        value = option.get("value")
        if isinstance(value, (int, float)) and float(value).is_integer():
            values.add(int(value))
    return values


def infer_scale_type(
    response_type: Any = None,
    response_options: Any = None,
    scale_guidance: Any = None,
    help_text: Any = None,
    question: Any = None,
    min_value: Any = None,
    max_value: Any = None,
) -> str:
    options = _normalize_option_list(response_options)
    values = _numeric_set(options)

    rt = normalize_text(response_type)
    labels_text = " ".join(normalize_text(opt.get("label")) for opt in options)
    context = " ".join(
        [
            rt,
            normalize_text(scale_guidance),
            normalize_text(help_text),
            normalize_text(question),
            labels_text,
        ]
    )

    temporal_terms = [
        "ultimos 6 meses",
        "ultimos seis meses",
        "12 meses",
        "ocurrio",
        "reciente",
        "antes",
        "no recientemente",
    ]
    observation_terms = ["se observa", "claramente", "hay duda", "persistente", "a veces"]
    frequency_terms = ["nunca", "ocasional", "frecuente", "casi siempre", "siempre", "todo el tiempo"]
    impact_terms = ["impacto", "afecta", "moderado", "marcado", "interfiere", "grave"]

    if _contains_any(rt, ["binary", "boolean", "yes_no", "si_no", "binaria"]):
        return "binary"
    if values == {0, 1} and _contains_any(context, [" si ", " no ", "si", "no", "verdadero", "falso"]):
        return "binary"

    if values == {0, 1, 2}:
        if _contains_any(context, temporal_terms):
            return "temporal_0_2"
        if _contains_any(context, observation_terms):
            return "observation_0_2"

    if values == {0, 1, 2, 3}:
        has_impact = _contains_any(context, impact_terms)
        has_frequency = _contains_any(context, frequency_terms)
        if has_impact and not has_frequency:
            return "impact_0_3"
        if has_frequency and not has_impact:
            return "frequency_0_3"
        if _contains_any(rt, ["impact"]):
            return "impact_0_3"
        if _contains_any(rt, ["frecuencia", "frequency"]):
            return "frequency_0_3"
        return "frequency_0_3"

    min_num = safe_float(min_value)
    max_num = safe_float(max_value)
    if min_num is not None and max_num is not None and _contains_any(
        rt,
        ["numeric", "number", "edad", "range", "int", "float"],
    ):
        return "numeric_range"

    if min_num is not None and max_num is not None and not options:
        return "numeric_range"

    if options:
        return "categorical"
    return "unknown"


def _sentence_from_labels(labels: List[str]) -> str:
    items = [item.strip() for item in labels if item and item.strip()]
    if not items:
        return "Puedes responder con tus palabras."
    if len(items) == 1:
        return f"Puedes responder: {items[0]}."
    if len(items) == 2:
        return f"Puedes responder: {items[0]} o {items[1]}."
    return f"Puedes responder: {', '.join(items[:-1])}, o {items[-1]}."


def _default_human_payload(scale_type: str) -> Dict[str, Any]:
    if scale_type == "binary":
        return {
            "human_options_text": "Puedes responder: No o Sí.",
            "quick_chips": ["Sí", "No", "No entiendo"],
        }
    if scale_type == "temporal_0_2":
        return {
            "human_options_text": "Puedes responder: no ocurrio, ocurrio antes, u ocurrio recientemente.",
            "quick_chips": ["No ocurrio", "Ocurrio antes", "Ocurrio recientemente", "No entiendo"],
        }
    if scale_type == "observation_0_2":
        return {
            "human_options_text": "Puedes responder: no se observa, a veces, o claramente.",
            "quick_chips": ["No se observa", "A veces", "Claramente", "No entiendo"],
        }
    if scale_type == "frequency_0_3":
        return {
            "human_options_text": "Puedes responder: nunca, leve u ocasional, frecuente, o casi siempre.",
            "quick_chips": ["Nunca", "Ocasional", "Frecuente", "Casi siempre", "No entiendo"],
        }
    if scale_type == "impact_0_3":
        return {
            "human_options_text": "Puedes responder: no afecta, leve, moderado, o marcado.",
            "quick_chips": ["No afecta", "Leve", "Moderado", "Marcado", "No entiendo"],
        }
    if scale_type == "numeric_range":
        return {
            "human_options_text": "Responde con un numero dentro del rango esperado.",
            "quick_chips": ["No entiendo"],
        }
    if scale_type == "categorical":
        return {
            "human_options_text": "Elige la opcion que mejor describa la situacion.",
            "quick_chips": ["No entiendo"],
        }
    return {
        "human_options_text": "Puedes responder con tus palabras.",
        "quick_chips": ["No entiendo"],
    }


def _dedupe_chips(chips: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for chip in chips:
        text = str(chip or "").strip()
        if not text:
            continue
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


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
    raw = response_options if response_options is not None else response_options_json
    options_list = _normalize_option_list(raw)

    scale_type = infer_scale_type(
        response_type=response_type,
        response_options=options_list,
        scale_guidance=scale_guidance,
        help_text=help_text,
        question=question,
        min_value=min_value,
        max_value=max_value,
    )

    payload = _default_human_payload(scale_type)

    if scale_type == "categorical" and options_list:
        labels = [str(opt.get("label") or opt.get("value") or "").strip() for opt in options_list]
        payload["human_options_text"] = _sentence_from_labels(labels)
        payload["quick_chips"] = _dedupe_chips(labels[:4] + ["No entiendo"])

    if scale_type == "binary" and options_list:
        labels = [normalize_text(opt.get("label")) for opt in options_list]
        if any(label.startswith("si") for label in labels) and any(label.startswith("no") for label in labels):
            payload["human_options_text"] = "Puedes responder: No o Sí."
            payload["quick_chips"] = ["Sí", "No", "No entiendo"]

    payload["quick_chips"] = _dedupe_chips(payload.get("quick_chips", []))

    return {
        "options_list": options_list,
        "human_options_text": payload["human_options_text"],
        "quick_chips": payload["quick_chips"],
        "scale_type": scale_type,
    }
