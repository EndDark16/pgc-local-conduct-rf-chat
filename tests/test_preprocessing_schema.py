import pandas as pd

from src.preprocessing import build_feature_schema, build_preprocessor, prepare_features_frame


def test_prepare_and_preprocessor():
    df = pd.DataFrame(
        {
            "age_years": [8, 9, None, 10],
            "sex_assigned_at_birth": ["M", "F", "M", None],
            "conduct_02_initiates_fights": [0, 1, 2, 1],
        }
    )
    selected = ["age_years", "sex_assigned_at_birth", "conduct_02_initiates_fights"]
    X = prepare_features_frame(df, selected)
    preprocessor, info = build_preprocessor(X)
    preprocessor.fit(X)
    Xt = preprocessor.transform(X)
    assert Xt.shape[0] == 4
    assert "numeric_columns" in info


def test_feature_schema_creation():
    selected = ["conduct_02_initiates_fights"]
    metadata = {
        "conduct_02_initiates_fights": {
            "feature": "conduct_02_initiates_fights",
            "questionnaire_item_id": "q1",
            "question_text_primary": "¿Inicia peleas físicas?",
            "caregiver_question": "¿Has observado que inicia peleas físicas?",
            "psychologist_question": "¿Inicia peleas físicas de forma activa?",
            "response_type": "temporal_012",
            "response_options_json": "[0,1,2]",
            "min_value": 0,
            "max_value": 2,
            "show_in_questionnaire_yes_no": "yes",
        }
    }
    schema = build_feature_schema(selected, metadata, persist=False)
    assert schema["feature_order"] == selected
    assert schema["features"][0]["feature"] == "conduct_02_initiates_fights"
    assert "parser_rules" in schema["features"][0]
