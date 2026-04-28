from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web_app import app


client = TestClient(app)


def _has_questionnaire() -> bool:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    return any(data_dir.glob("questionnaire*.csv"))


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
