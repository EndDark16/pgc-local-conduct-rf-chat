from src.nlp_interpreter import (
    detect_user_does_not_understand,
    interpret_answer,
    is_help_request,
    normalize_text,
)


BINARY_META = {
    "response_type": "yes_no",
    "response_options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Sí"}],
    "min_value": 0,
    "max_value": 1,
}

FREQUENCY_META = {
    "response_type": "frequency_0_3",
    "response_options": [
        {"value": 0, "label": "Nunca"},
        {"value": 1, "label": "Leve u ocasional"},
        {"value": 2, "label": "Claro o frecuente"},
        {"value": 3, "label": "Muy marcado o casi siempre"},
    ],
    "min_value": 0,
    "max_value": 3,
}

TEMPORAL_META = {
    "response_type": "temporal_0_2",
    "response_options": [
        {"value": 0, "label": "No ocurrió"},
        {"value": 1, "label": "Ocurrió en los últimos 12 meses, pero no en los últimos 6"},
        {"value": 2, "label": "Ocurrió en los últimos 6 meses"},
    ],
    "min_value": 0,
    "max_value": 2,
}

IMPACT_META = {
    "response_type": "impact_0_3",
    "response_options": [
        {"value": 0, "label": "Sin impacto"},
        {"value": 1, "label": "Leve"},
        {"value": 2, "label": "Moderado"},
        {"value": 3, "label": "Marcado"},
    ],
    "min_value": 0,
    "max_value": 3,
}


def test_normalize_text():
    assert normalize_text("sí suele pasar") == "si suele pasar"
    assert normalize_text("a veces") == "a veces"


def test_help_request_detection():
    assert is_help_request("No entiendo la pregunta")
    assert detect_user_does_not_understand("no se que responder")


def test_binary_yes_phrase_maps_to_1():
    result = interpret_answer("x", "si suele pasar", BINARY_META)
    assert result["parsed_value"] == 1
    assert result["needs_clarification"] is False


def test_binary_a_veces_requests_clarification():
    result = interpret_answer("x", "a veces", BINARY_META)
    assert result["parsed_value"] is None
    assert result["needs_clarification"] is True
    assert "sí" in result["clarification_question"] or "si" in result["clarification_question"]


def test_frequency_casi_siempre_maps_to_3():
    result = interpret_answer("x", "casi siempre", FREQUENCY_META)
    assert result["parsed_value"] == 3
    assert result["needs_clarification"] is False


def test_frequency_recently_requires_frequency_clarification():
    result = interpret_answer("x", "ocurrió recientemente", FREQUENCY_META)
    assert result["parsed_value"] is None
    assert result["needs_clarification"] is True
    assert "frecu" in normalize_text(result["clarification_question"])


def test_temporal_recently_maps_to_2():
    result = interpret_answer("conduct_02_initiates_fights", "ocurrió recientemente", TEMPORAL_META)
    assert result["parsed_value"] == 2
    assert result["needs_clarification"] is False


def test_impact_parser():
    result = interpret_answer("conduct_impairment_global", "impacto moderado", IMPACT_META)
    assert result["parsed_value"] == 2


def test_no_entiendo_category():
    result = interpret_answer("x", "no entiendo", BINARY_META)
    assert result["parsed_value"] is None
    assert normalize_text(result["answer_category"]).startswith("usuario no entend")
