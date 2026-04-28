from pathlib import Path


def test_basic_files_exist():
    root = Path(__file__).resolve().parent.parent
    required = [
        root / "app.py",
        root / "train.py",
        root / "src" / "web_app.py",
        root / "web" / "templates" / "index.html",
        root / "web" / "static" / "css" / "styles.css",
        root / "web" / "static" / "js" / "app.js",
    ]
    for path in required:
        assert path.exists(), f"Missing file: {path}"
