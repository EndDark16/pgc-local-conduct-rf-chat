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

OBS_META = {
    "response_type": "observation_0_2",
    "response_options": [
        {"value": 0, "label": "No se observa"},
        {"value": 1, "label": "Se observa a veces"},
        {"value": 2, "label": "Se observa claramente"},
    ],
    "min_value": 0,
    "max_value": 2,
}

NUM_META = {
    "response_type": "numeric_range",
    "min_value": 5,
    "max_value": 18,
}


def test_normalize_text():
    assert normalize_text("sí suele pasar") == "si suele pasar"
    assert normalize_text("a veses") == "a veces"


def test_help_request_detection_strict():
    assert is_help_request("No entiendo la pregunta")
    assert detect_user_does_not_understand("no se que responder")
    assert not is_help_request("no se observa")


def test_binary_yes_phrase_maps_to_1():
    result = interpret_answer("x", "si suele pasar", BINARY_META)
    assert result["parsed_value"] == 1
    assert result["needs_clarification"] is False


def test_binary_a_veces_requests_clarification():
    result = interpret_answer("x", "a veces", BINARY_META)
    assert result["parsed_value"] is None
    assert result["needs_clarification"] is True
    assert "si" in normalize_text(result["clarification_question"])


def test_temporal_partial_occurrence_requests_only_time():
    result = interpret_answer("x", "si ocurrio", TEMPORAL_META)
    assert result["parsed_value"] is None
    assert result["needs_clarification"] is True
    clarification = normalize_text(result["clarification_question"])
    assert "cuando" in clarification or "antes" in clarification or "anterior" in clarification


def test_frequency_casi_siempre_maps_to_3():
    result = interpret_answer("x", "casi siempre", FREQUENCY_META)
    assert result["parsed_value"] == 3


def test_frequency_todo_el_tiempo_maps_to_3():
    result = interpret_answer("x", "todo el tiempo", FREQUENCY_META)
    assert result["parsed_value"] == 3


def test_frequency_recently_requires_frequency_clarification():
    result = interpret_answer("x", "ocurrió recientemente", FREQUENCY_META)
    assert result["parsed_value"] is None
    assert result["needs_clarification"] is True
    assert "frecu" in normalize_text(result["clarification_question"])


def test_temporal_recently_maps_to_2():
    result = interpret_answer("x", "ocurrió recientemente", TEMPORAL_META)
    assert result["parsed_value"] == 2


def test_observation_no_se_observa_maps_to_0():
    result = interpret_answer("x", "no se observa", OBS_META)
    assert result["parsed_value"] == 0


def test_impact_natural_phrase_maps_to_3():
    result = interpret_answer("x", "afecta mucho en casa y en el colegio", IMPACT_META)
    assert result["parsed_value"] == 3


def test_numeric_value_extraction_inside_text():
    result = interpret_answer("age_years", "tiene 10 años", NUM_META)
    assert result["parsed_value"] == 10


def test_no_entiendo_category():
    result = interpret_answer("x", "no entiendo", BINARY_META)
    assert result["parsed_value"] is None
    assert normalize_text(result["answer_category"]).startswith("usuario no entend")
