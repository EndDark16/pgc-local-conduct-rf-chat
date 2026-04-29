import pandas as pd

from src.leakage_audit import run_leakage_audit


def test_leakage_audit_detects_name_patterns_and_persists_false():
    df = pd.DataFrame(
        {
            "conduct_01_bullies_threatens_intimidates": [0, 0, 1, 1, 2, 2, 1, 0],
            "conduct_total_score": [0, 1, 2, 3, 4, 5, 4, 3],
            "age_years": [8, 9, 10, 11, 12, 13, 14, 15],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1, 1, 0])
    report = run_leakage_audit(
        dataset_df=df,
        y_binary=y,
        candidate_features=list(df.columns),
        target_col="target_domain_conduct_final",
        persist=False,
    )

    excluded_features = {item["feature"] for item in report["columns_excluded"]}
    assert "conduct_total_score" in excluded_features
    assert report["summary"]["excluded_count"] >= 1
    assert report["reviewed_count"] == 3
