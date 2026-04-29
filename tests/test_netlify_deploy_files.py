from pathlib import Path


def test_netlify_files_exist_and_have_required_rules():
    root = Path(__file__).resolve().parent.parent
    netlify_toml = root / "netlify.toml"
    web_index = root / "web" / "index.html"
    proxy_fn = root / "netlify" / "functions" / "api-proxy.js"

    assert netlify_toml.exists(), "Missing netlify.toml"
    assert web_index.exists(), "Missing web/index.html for Netlify publish dir"
    assert proxy_fn.exists(), "Missing Netlify API proxy function"

    toml_text = netlify_toml.read_text(encoding="utf-8", errors="ignore")
    assert 'publish = "web"' in toml_text
    assert 'functions = "netlify/functions"' in toml_text
    assert 'from = "/api/*"' in toml_text
    assert 'to = "/.netlify/functions/api-proxy/:splat"' in toml_text
    assert 'from = "/*"' in toml_text
    assert 'to = "/index.html"' in toml_text

    index_text = web_index.read_text(encoding="utf-8", errors="ignore")
    assert '/static/css/styles.css' in index_text
    assert '/static/js/app.js' in index_text

    proxy_text = proxy_fn.read_text(encoding="utf-8", errors="ignore")
    assert "BACKEND_API_URL" in proxy_text
    assert "/api/" in proxy_text
    assert "netlify_local_fallback" in proxy_text
    assert "Conexión de backend pendiente" not in proxy_text
