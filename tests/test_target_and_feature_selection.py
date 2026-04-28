from pathlib import Path

import pytest

from src.config import DEFAULT_TARGET_COLUMN, resolve_target_column
from src.data_loader import load_main_dataset
from src.feature_selection import select_features
from src.questionnaire_loader import load_questionnaire


def _has_data_files() -> bool:
    root = Path(__file__).resolve().parent.parent / "data"
    return any(root.glob("hybrid_no_external_scores_dataset_ready*.csv")) and any(
        root.glob("questionnaire*.csv")
    )


def test_target_resolution_default():
    target, method = resolve_target_column(["target_domain_conduct_final", "target_domain_anxiety_final"])
    assert target == DEFAULT_TARGET_COLUMN
    assert method in {"default_conduct", "fallback_first_available_target", "TARGET_DISORDER=conduct"}


@pytest.mark.skipif(not _has_data_files(), reason="No data files in data/")
def test_select_features_conduct_excludes_adhd_prefix():
    df, _ = load_main_dataset()
    q_df, _, _ = load_questionnaire()
    report = select_features(df, q_df, target_col="target_domain_conduct_final")
    assert report["features_used_count"] > 0
    assert "target_domain_conduct_final" not in report["features_used"]
    assert all(not f.startswith("target_domain_") for f in report["features_used"])
    assert all(not f.startswith("adhd_") for f in report["features_used"])
