"""Shared utility helpers."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

try:
    from unidecode import unidecode as _unidecode
except Exception:  # pragma: no cover - fallback when dependency is unavailable
    _unidecode = None


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
    except Exception:
        pass

    if isinstance(value, Path):
        return str(value)
    return str(value)


def normalize_text(value: Any) -> str:
    """Lowercase + accent stripping + spaces normalization for robust matching."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    if _unidecode is not None:
        text = _unidecode(text)
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_column_name(name: str) -> str:
    """Normalize column names to snake_case ASCII."""
    base = normalize_text(name)
    base = base.replace("/", "_").replace("-", "_")
    base = re.sub(r"[^a-z0-9_ ]+", "", base)
    base = base.replace(" ", "_")
    base = re.sub(r"_+", "_", base)
    return base.strip("_")


def hash_file(path: Path) -> str:
    """SHA256 hash for file content."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dataframe(df: pd.DataFrame) -> str:
    """Stable hash based on dataframe values and columns."""
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_bool(value: Any) -> bool:
    text = normalize_text(value)
    return text in {"1", "true", "yes", "y", "si", "sí", "s", "ok"}


def safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_existing_path(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def try_parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # A permissive fallback for single-quoted JSON-like strings.
        normalized = text.replace("'", '"')
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            return None
