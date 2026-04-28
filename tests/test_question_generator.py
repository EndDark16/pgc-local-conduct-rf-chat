from src.question_generator import question_for_feature


def test_question_generator_prefers_role_specific_text_and_human_options():
    meta = {
        "feature": "conduct_01_bullies_threatens_intimidates",
        "caregiver_question": "¿Ha intimidado a otros niños?",
        "psychologist_question": "¿Se observan conductas de intimidación hacia pares?",
        "question_text_primary": "Pregunta base",
        "help_text": "Observa situaciones en casa y colegio.",
        "scale_guidance": "0 no ocurrió, 1 ocurrió antes, 2 ocurrió reciente.",
        "response_type": "temporal_0_2",
        "response_options": [
            {"value": 0, "label": "No ocurrió"},
            {"value": 1, "label": "Ocurrió antes"},
            {"value": 2, "label": "Ocurrió recientemente"},
        ],
    }
    q1 = question_for_feature(meta, role="caregiver")
    q2 = question_for_feature(meta, role="psychologist")
    assert "intimidado" in q1["question"].lower()
    assert "intimidación" in q2["question"].lower()
    assert "human_options_text" in q1
    assert "No entiendo" in q1["quick_chips"]
