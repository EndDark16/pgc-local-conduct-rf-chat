"""Project setup helper: create directories and verify dependencies."""

from __future__ import annotations

import importlib.util
import platform
from typing import Dict, List

from src.config import REQUIRED_DIRS, ensure_required_dirs


DEPENDENCIES = [
    "pandas",
    "numpy",
    "sklearn",
    "joblib",
    "fastapi",
    "uvicorn",
    "jinja2",
    "multipart",
    "matplotlib",
    "rapidfuzz",
    "unidecode",
    "openpyxl",
    "pytest",
    "docx",
]


def main() -> int:
    ensure_required_dirs()
    print("Sistema:", platform.system(), platform.release())
    print("Carpetas verificadas/creadas:")
    for directory in REQUIRED_DIRS:
        print(f"- {directory}")

    missing: List[str] = []
    status: Dict[str, bool] = {}
    for dep in DEPENDENCIES:
        found = importlib.util.find_spec(dep) is not None
        status[dep] = found
        if not found:
            missing.append(dep)

    print("\nDependencias:")
    for dep, found in status.items():
        print(f"- {dep}: {'OK' if found else 'FALTA'}")

    if missing:
        print("\nFaltan dependencias. Instala con:")
        print("pip install -r requirements.txt")
    else:
        print("\nEntorno listo para entrenamiento, validación, pruebas y app local.")
    print("Este proyecto corre en CPU y no requiere GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
