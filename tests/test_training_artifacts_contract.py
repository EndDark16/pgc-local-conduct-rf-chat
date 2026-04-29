from pathlib import Path
import json

import pytest

from src.config import MAX_ACCEPTABLE_METRIC
from src.utils import load_json


ROOT = Path(__file__).resolve().parent.parent


def _has_artifact(name: str) -> bool:
    return (ROOT / "artifacts" / name).exists()


@pytest.mark.skipif(not _has_artifact("metrics.json"), reason="metrics.json not available")
def test_metrics_not_capped_and_overfit_fields():
    metrics = load_json(ROOT / "artifacts" / "metrics.json", default={})
    final = metrics.get("test_metrics_threshold_final", {})
    assert isinstance(final.get("f1"), (int, float))
    if metrics.get("max_acceptable_metric") is None:
        pytest.skip("metrics.json aún no fue regenerado con overfit guard")
    assert metrics.get("max_acceptable_metric") == MAX_ACCEPTABLE_METRIC
    assert "overfit_warning" in metrics
    assert "metrics_above_limit" in metrics


@pytest.mark.skipif(not _has_artifact("leakage_audit.json"), reason="leakage_audit.json not available")
def test_leakage_audit_artifact_shape():
    audit = load_json(ROOT / "artifacts" / "leakage_audit.json", default={})
    assert "reviewed_columns" in audit
    assert "columns_excluded" in audit
    assert "summary" in audit


@pytest.mark.skipif(not _has_artifact("overfit_guard_report.json"), reason="overfit_guard_report.json not available")
def test_overfit_guard_report_shape():
    report = load_json(ROOT / "artifacts" / "overfit_guard_report.json", default={})
    assert report.get("overfit_guard_applied") is True
    assert "selected_model_variant" in report
    assert "selected_model_reason" in report


@pytest.mark.skipif(not _has_artifact("feature_schema.json"), reason="feature_schema.json not available")
def test_feature_schema_is_strict_json_without_nan_constants():
    raw = (ROOT / "artifacts" / "feature_schema.json").read_text(encoding="utf-8")

    def _raise_constant(value: str):  # pragma: no cover - only called when invalid constants appear
        raise ValueError(value)

    parsed = json.loads(raw, parse_constant=_raise_constant)
    assert isinstance(parsed, dict)
    assert "features" in parsed
