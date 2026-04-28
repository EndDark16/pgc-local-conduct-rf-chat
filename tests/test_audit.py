from pathlib import Path

from src.audit import AUDIT_LOG_PATH, audit_event


def test_audit_write():
    before = AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines() if AUDIT_LOG_PATH.exists() else []
    audit_event("test_event", {"ok": True})
    after = AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines() if AUDIT_LOG_PATH.exists() else []
    assert len(after) >= len(before) + 1
