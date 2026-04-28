from src.question_explainer import explain_question


def test_question_explanation_structure_and_human_output():
    meta = {
        "feature_label_human": "Inicia peleas",
        "feature_description": "Evalúa si inicia peleas físicas.",
        "help_text": "Considera si ha iniciado golpes o empujones.",
        "scale_guidance": "0 no ocurrió, 1 ocurrió antes, 2 ocurrió reciente.",
        "response_type": "temporal_0_2",
        "response_options": [
            {"value": 0, "label": "No ocurrió"},
            {"value": 1, "label": "Ocurrió antes"},
            {"value": 2, "label": "Ocurrió recientemente"},
        ],
    }
    result = explain_question("conduct_02_initiates_fights", meta)
    assert result["feature_name"] == "conduct_02_initiates_fights"
    assert "simple_explanation" in result and result["simple_explanation"]
    assert isinstance(result["examples"], list) and result["examples"]
    assert "expected_answer" in result
    assert "human_options_text" in result
    assert "quick_chips" in result and "No entiendo" in result["quick_chips"]


def test_explanation_avoids_technical_input_for_text():
    meta = {
        "question_text_primary": "Input for context - Adhd context home",
        "help_text": "Input for context - adhd_hypimp_01_fidgets",
        "response_type": "yes_no",
        "response_options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Sí"}],
    }
    result = explain_question("conduct_02_initiates_fights", meta)
    text = (result["simple_explanation"] + " " + result["expected_answer"]).lower()
    assert "input for" not in text
    assert "adhd_hypimp_01_fidgets" not in text
