from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web_app import app


client = TestClient(app)


def _has_questionnaire() -> bool:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    return any(data_dir.glob("questionnaire*.csv"))


def _has_model() -> bool:
    root = Path(__file__).resolve().parent.parent
    return (root / "models" / "model.joblib").exists() and (root / "models" / "preprocessor.joblib").exists()


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_model_status_endpoint():
    response = client.get("/api/model-status")
    assert response.status_code == 200
    payload = response.json()
    assert "model_trained" in payload
    assert "target_column" in payload


@pytest.mark.skipif(not _has_questionnaire(), reason="No questionnaire file in data/")
def test_questions_interpret_explain_endpoints_human_and_conduct_safe():
    response = client.get("/api/questions?role=caregiver&session_id=testcase")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert "intro_text" in data

    if data.get("target_column") == "target_domain_conduct_final":
        assert all(not q["feature"].startswith("adhd_") for q in data["questions"])

    first_q = data["questions"][0]
    assert first_q["question"]
    assert "human_options_text" in first_q
    assert "quick_chips" in first_q
    assert "[{\"value\"" not in first_q.get("human_options_text", "")

    interp = client.post(
        "/api/chat/interpret",
        json={
            "feature": first_q["feature"],
            "answer": "si suele pasar",
            "role": "caregiver",
            "session_id": "testcase",
        },
    )
    assert interp.status_code == 200
    interp_data = interp.json()
    assert "needs_explanation" in interp_data

    explain = client.post(
        "/api/chat/explain",
        json={
            "feature": first_q["feature"],
            "mode": "simple",
        },
    )
    assert explain.status_code == 200
    explain_data = explain.json()
    assert "simple_explanation" in explain_data
    assert "Input for" not in explain_data["simple_explanation"]


def test_result_question_requires_prediction():
    response = client.post(
        "/api/chat/result-question",
        json={"session_id": "new-session-no-result", "question": "que significa este resultado"},
    )
    assert response.status_code == 400


def test_metrics_endpoint_confusion_payload_shape():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert "ok" in payload
    if payload.get("ok"):
        assert "confusion_matrix_detail" in payload
        detail = payload["confusion_matrix_detail"]
        assert "matrix" in detail
        assert {"tn", "fp", "fn", "tp"}.issubset(detail.keys())


@pytest.mark.skipif(not (_has_questionnaire() and _has_model()), reason="No questionnaire/model available")
def test_can_confirm_all_questions_and_predict_result():
    session = "flow-complete-test"
    response = client.get(f"/api/questions?role=caregiver&session_id={session}")
    assert response.status_code == 200
    payload = response.json()
    questions = payload["questions"]
    assert questions

    answers = {}
    for q in questions:
        feature = q["feature"]
        options = q.get("response_options") or []
        value = None
        if options:
            first = options[0]
            value = first.get("value")
            if value is None:
                value = first.get("label")
        if value is None:
            value = q.get("min_value")
        if value is None:
            value = 0

        confirm = client.post(
            "/api/chat/confirm",
            json={
                "feature": feature,
                "parsed_value": value,
                "raw_answer": str(value),
                "confidence": 0.8,
                "session_id": session,
            },
        )
        assert confirm.status_code == 200, confirm.text
        answers[feature] = value

    pred = client.post("/api/predict", json={"session_id": session, "answers": answers})
    assert pred.status_code == 200
    pred_payload = pred.json()
    assert pred_payload.get("ok") is True
    assert "prediction" in pred_payload
