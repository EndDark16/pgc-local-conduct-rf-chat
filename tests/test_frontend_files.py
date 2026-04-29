from pathlib import Path


def test_frontend_files_exist_and_not_empty():
    root = Path(__file__).resolve().parent.parent
    html = root / "web" / "templates" / "index.html"
    css = root / "web" / "static" / "css" / "styles.css"
    js = root / "web" / "static" / "js" / "app.js"
    for path in [html, css, js]:
        assert path.exists(), f"Missing frontend file: {path}"
        assert path.stat().st_size > 10, f"Frontend file appears empty: {path}"


def test_frontend_chat_contract_and_no_forbidden_texts():
    root = Path(__file__).resolve().parent.parent
    html = (root / "web" / "templates" / "index.html").read_text(encoding="utf-8", errors="ignore")
    css = (root / "web" / "static" / "css" / "styles.css").read_text(encoding="utf-8", errors="ignore")
    js = (root / "web" / "static" / "js" / "app.js").read_text(encoding="utf-8", errors="ignore")

    assert "/static/css/styles.css" in html
    assert "/static/js/app.js" in html

    for required in [
        "app-shell",
        "chat-panel",
        "chat-messages",
        "quick-actions",
        "chat-input-form",
        "technicalDetails",
        "metricsChart",
        "confusionChart",
        "importanceChart",
    ]:
        assert required in html

    forbidden = [
        "Cargar preguntas",
        "Pregunta 0 de 0",
        "Pulsa \"Cargar preguntas\"",
        "Pulsa Cargar preguntas",
        "Input for",
        "response_options_json",
        "Actualizar métricas",
        "Actualizar importancia",
        "loadMetricsBtn",
        "loadImportanceBtn",
    ]
    for token in forbidden:
        assert token not in html
        assert token not in js

    # no raw JSON options rendered in html/js literals for user text
    for token in ["[{'value':", '[{"value": 0, "label": "No"}']:
        assert token not in html
        assert token not in js

    for color in ["#080b12", "#101624", "#151c2e", "#27324a", "#7c5cff", "#22d3ee"]:
        assert color in css.lower()

    assert "initializeChatFlow" in js
    assert "/api/questions" in js
    assert "/api/chat/result-question" in js
    assert "STATE.RESULT_QA" in js
    assert 'technicalDetails.addEventListener("toggle"' in js
    assert "Generar impresión orientativa" in js
