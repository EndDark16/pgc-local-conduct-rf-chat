"""Local audit logging utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import LOGS_DIR


AUDIT_LOG_PATH = LOGS_DIR / "audit.jsonl"


def audit_event(event: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Append one structured audit event to logs/audit.jsonl."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload or {},
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def audit_error(event: str, error: Exception, payload: Dict[str, Any] | None = None) -> None:
    full_payload = dict(payload or {})
    full_payload.update(
        {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )
    audit_event(event, full_payload)


def load_audit_tail(limit: int = 50) -> list[dict]:
    path = Path(AUDIT_LOG_PATH)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:] if line.strip()]
