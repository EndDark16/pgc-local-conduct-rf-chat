from src.response_options import format_response_options


def test_binary_options_are_humanized():
    payload = format_response_options(
        response_options=[{"value": 0, "label": "No"}, {"value": 1, "label": "Sí"}],
        response_type="yes_no",
    )
    assert payload["scale_type"] == "binary"
    assert "No o Sí" in payload["human_options_text"]
    assert "No entiendo" in payload["quick_chips"]


def test_frequency_scale_human_text_and_chips():
    payload = format_response_options(
        response_options=[
            {"value": 0, "label": "Nunca"},
            {"value": 1, "label": "Leve"},
            {"value": 2, "label": "Frecuente"},
            {"value": 3, "label": "Casi siempre"},
        ],
        response_type="frequency_0_3",
    )
    assert payload["scale_type"] == "frequency_0_3"
    assert "casi siempre" in payload["human_options_text"].lower()
    assert "Frecuente" in payload["quick_chips"]

