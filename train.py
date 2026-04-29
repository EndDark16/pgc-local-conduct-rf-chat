"""Training pipeline for local Random Forest model (PGC project)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.audit import audit_error, audit_event
from src.config import (
    ARTIFACTS_DIR,
    CONDUCT_SECTION_NAME,
    MODELS_DIR,
    TARGET_COLUMNS,
    ensure_required_dirs,
    resolve_target_column,
)
from src.data_contract import analyze_data_contract
from src.data_loader import load_main_dataset
from src.feature_selection import (
    DEFAULT_CONDUCT_FEATURES,
    _should_exclude_col,
    is_feature_allowed_for_target,
    select_features,
)
from src.model import build_random_forest, hyperparameter_space
from src.preprocessing import (
    build_feature_schema,
    build_preprocessor,
    prepare_features_frame,
    save_preprocessor,
)
from src.questionnaire_loader import load_questionnaire
from src.training_utils import evaluate_binary_metrics, threshold_search
from src.utils import as_bool, hash_file, normalize_text, save_json


def _safe_stratify_vector(y: pd.Series) -> pd.Series | None:
    counts = y.value_counts(dropna=False)
    if len(counts) < 2:
        return None
    if counts.min() < 2:
        return None
    return y


def _split_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    split_meta: Dict[str, Any] = {}

    strat_1 = _safe_stratify_vector(y)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=strat_1,
    )
    split_meta["first_split_stratified"] = strat_1 is not None

    strat_2 = _safe_stratify_vector(y_train_val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=42,
        stratify=strat_2,
    )
    split_meta["second_split_stratified"] = strat_2 is not None
    split_meta["sizes"] = {
        "train": len(X_train),
        "validation": len(X_val),
        "test": len(X_test),
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, split_meta


def _split_indices(y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    indices = np.arange(len(y))
    split_meta: Dict[str, Any] = {}
    strat_1 = _safe_stratify_vector(y)
    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=strat_1,
    )
    split_meta["first_split_stratified"] = strat_1 is not None

    y_train_val = y.iloc[idx_train_val]
    strat_2 = _safe_stratify_vector(y_train_val)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=0.20,
        random_state=42,
        stratify=strat_2,
    )
    split_meta["second_split_stratified"] = strat_2 is not None
    split_meta["sizes"] = {
        "train": int(len(idx_train)),
        "validation": int(len(idx_val)),
        "test": int(len(idx_test)),
    }
    return idx_train, idx_val, idx_test, split_meta


def _build_feature_set_candidates(
    dataset_df: pd.DataFrame,
    questionnaire_df: pd.DataFrame,
    base_features: List[str],
    target_col: str,
) -> List[Dict[str, Any]]:
    dataset_cols = set(dataset_df.columns)
    q_rows = {
        str(row["feature"]): row.to_dict()
        for _, row in questionnaire_df.iterrows()
        if str(row.get("feature", "")).strip()
    }

    candidates: List[Dict[str, Any]] = []
    candidates.append(
        {
            "name": "base_contract_selection",
            "description": "Selección base del contrato de datos.",
            "features": sorted(set(base_features)),
        }
    )

    strict_core: List[str] = []
    for feat in DEFAULT_CONDUCT_FEATURES:
        if feat in dataset_cols and feat in q_rows:
            exclude, _ = _should_exclude_col(feat, target_col)
            if not exclude:
                strict_core.append(feat)
    candidates.append(
        {
            "name": "strict_conduct_core",
            "description": "Features núcleo de conducta + edad/sexo.",
            "features": sorted(set(strict_core)),
        }
    )

    target_contract: List[str] = []
    section_match = normalize_text(CONDUCT_SECTION_NAME)
    for feat, row in q_rows.items():
        if feat not in dataset_cols:
            continue
        exclude, _ = _should_exclude_col(feat, target_col)
        if exclude:
            continue
        allowed, _ = is_feature_allowed_for_target(feat, row, target_col)
        if not allowed:
            continue
        section_name = normalize_text(row.get("section_name", ""))
        is_direct = as_bool(row.get("is_direct_input")) or as_bool(
            row.get("visible_question_yes_no")
        )
        if is_direct or section_name == section_match:
            target_contract.append(feat)
    for demo in ("age_years", "sex_assigned_at_birth"):
        if demo in dataset_cols and demo in q_rows:
            target_contract.append(demo)
    candidates.append(
        {
            "name": "target_domain_contract_inputs",
            "description": "Inputs directos del cuestionario filtrados por dominio del target + demograficos.",
            "features": sorted(set(target_contract)),
        }
    )

    # Cleanup: remove weak candidates and deduplicate by feature tuple.
    cleaned: List[Dict[str, Any]] = []
    seen_signatures = set()
    for cand in candidates:
        features = [
            f
            for f in cand["features"]
            if f in dataset_cols and not _should_exclude_col(f, target_col)[0]
        ]
        if len(features) < 5:
            continue
        signature = tuple(sorted(features))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        cleaned.append({**cand, "features": sorted(set(features)), "n_features": len(features)})

    if not cleaned:
        raise ValueError("No se pudieron construir candidatos de features para optimización.")
    return cleaned


def _is_better_candidate(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Return True when candidate `a` ranks above `b` by validation objective."""
    key_a = (
        float(a.get("model_selection_score", 0.0)),
        float(a["validation_metrics_final"]["f1"]),
        float(a["validation_metrics_final"]["recall"]),
        float(a["validation_metrics_final"]["precision"]),
    )
    key_b = (
        float(b.get("model_selection_score", 0.0)),
        float(b["validation_metrics_final"]["f1"]),
        float(b["validation_metrics_final"]["recall"]),
        float(b["validation_metrics_final"]["precision"]),
    )
    return key_a > key_b


def _pick_best_cv_index(search: RandomizedSearchCV) -> Tuple[int, Dict[str, Any]]:
    cv = search.cv_results_
    n_candidates = len(cv["params"])
    best_idx = 0
    best_score = -1e18
    diagnostics: Dict[str, Any] = {}

    for idx in range(n_candidates):
        mean_test_f1 = float(cv["mean_test_f1"][idx])
        mean_test_recall = float(cv["mean_test_recall"][idx])
        mean_test_precision = float(cv["mean_test_precision"][idx])
        std_test_f1 = float(cv.get("std_test_f1", [0.0] * n_candidates)[idx])
        mean_train_f1 = float(cv.get("mean_train_f1", cv["mean_test_f1"])[idx])
        overfit_gap = max(0.0, mean_train_f1 - mean_test_f1)

        # Penalize overfit and instability while prioritizing F1 and then recall.
        selection_score = (
            mean_test_f1
            + 0.18 * mean_test_recall
            + 0.03 * mean_test_precision
            - 0.60 * overfit_gap
            - 0.05 * std_test_f1
        )
        if selection_score > best_score:
            best_score = selection_score
            best_idx = idx
            diagnostics = {
                "mean_test_f1": mean_test_f1,
                "mean_test_recall": mean_test_recall,
                "mean_test_precision": mean_test_precision,
                "mean_train_f1": mean_train_f1,
                "overfit_gap_cv": overfit_gap,
                "std_test_f1": std_test_f1,
                "selection_score": selection_score,
            }

    return best_idx, diagnostics


def _plot_threshold_curve(rows: list[dict], output_path: Path) -> None:
    thresholds = [r["threshold"] for r in rows]
    f1_vals = [r["f1"] for r in rows]
    recall_vals = [r["recall"] for r in rows]
    precision_vals = [r["precision"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_vals, label="F1", linewidth=2)
    plt.plot(thresholds, recall_vals, label="Recall", linewidth=2)
    plt.plot(thresholds, precision_vals, label="Precision", linewidth=2)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Curva de threshold en validación")
    plt.grid(alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_confusion_matrix(cm: list[list[int]], output_path: Path) -> None:
    matrix = np.array(cm)
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Matriz de confusión (test, threshold final)")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["Real 0", "Real 1"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _extract_feature_importance(
    fitted_pipeline: Pipeline,
    feature_schema: Dict[str, Any],
) -> Dict[str, Any]:
    model = fitted_pipeline.named_steps["model"]
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out().tolist()
    importances = model.feature_importances_.tolist()
    cat_features: list[str] = []
    num_features: list[str] = []
    for name, _transformer, columns in preprocessor.transformers_:
        if name == "cat":
            cat_features = [str(c) for c in columns]
        if name == "num":
            num_features = [str(c) for c in columns]

    cat_features_sorted = sorted(cat_features, key=len, reverse=True)

    raw_items = [
        {"transformed_feature": feat, "importance": float(imp)}
        for feat, imp in zip(feature_names, importances)
    ]
    raw_items.sort(key=lambda x: x["importance"], reverse=True)

    # Aggregate by base feature before one-hot suffix.
    aggregated: Dict[str, float] = {}
    for item in raw_items:
        transformed = item["transformed_feature"]
        if transformed.startswith("cat__"):
            candidate = transformed.split("cat__", 1)[1]
            base = candidate
            for original in cat_features_sorted:
                if candidate == original or candidate.startswith(original + "_"):
                    base = original
                    break
        elif transformed.startswith("num__"):
            candidate = transformed.split("num__", 1)[1]
            base = candidate if candidate in num_features else candidate
        else:
            base = transformed
        aggregated[base] = aggregated.get(base, 0.0) + item["importance"]

    metadata_map = {f["feature"]: f for f in feature_schema.get("features", [])}
    agg_items = [
        {
            "feature": feat,
            "importance": float(score),
            "label": metadata_map.get(feat, {}).get("feature_label_human") or feat,
            "question": metadata_map.get(feat, {}).get("caregiver_question")
            or metadata_map.get(feat, {}).get("question_text_primary"),
        }
        for feat, score in aggregated.items()
    ]
    agg_items.sort(key=lambda x: x["importance"], reverse=True)
    return {
        "top_transformed_features": raw_items[:50],
        "feature_importance_aggregated": agg_items,
    }


def _plot_feature_importance(agg_items: list[dict], output_path: Path, top_n: int = 20) -> None:
    top = agg_items[:top_n]
    labels = [item["feature"] for item in top][::-1]
    values = [item["importance"] for item in top][::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(labels, values)
    plt.title("Importancia de variables (Random Forest)")
    plt.xlabel("Importancia")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _normalize_target_binary(y: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    y_no_na = y.dropna()
    unique_vals = sorted(y_no_na.unique().tolist())
    if len(unique_vals) != 2:
        raise ValueError(
            "El target no es binario o no tiene dos clases válidas. "
            f"Valores encontrados: {unique_vals}"
        )
    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
    return y.map(mapping), {str(k): v for k, v in mapping.items()}


def _max_primary_metric(metrics: Dict[str, Any]) -> float:
    values = [
        float(metrics.get("accuracy") or 0.0),
        float(metrics.get("precision") or 0.0),
        float(metrics.get("recall") or 0.0),
        float(metrics.get("f1") or 0.0),
    ]
    return max(values) if values else 0.0


def _build_generalization_alerts(
    train_metrics: Dict[str, Any],
    validation_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
) -> List[str]:
    alerts: List[str] = []
    max_test = _max_primary_metric(test_metrics)
    if max_test > 0.98:
        alerts.append(
            "Las métricas de test superan 0.98. Esto puede indicar señales muy directas del target "
            "o la necesidad de validación externa adicional."
        )

    train_test_gap_f1 = float(train_metrics.get("f1") or 0.0) - float(test_metrics.get("f1") or 0.0)
    if train_test_gap_f1 > 0.08:
        alerts.append(
            "La brecha F1 train-test es alta (>0.08). Conviene reforzar regularización o revisar features."
        )

    val_test_gap_f1 = abs(float(validation_metrics.get("f1") or 0.0) - float(test_metrics.get("f1") or 0.0))
    if val_test_gap_f1 > 0.06:
        alerts.append(
            "La brecha F1 validación-test es alta (>0.06). Se recomienda revisión adicional de generalización."
        )

    return alerts


def main() -> int:
    ensure_required_dirs()
    audit_event("training_started", {})

    try:
        contract_report = analyze_data_contract()
        dataset_df, dataset_profile = load_main_dataset()
        questionnaire_df, questionnaire_profile, feature_to_metadata = load_questionnaire()

        target_col, target_method = resolve_target_column(
            available_targets=[c for c in dataset_df.columns if c in TARGET_COLUMNS]
        )
        save_json(
            ARTIFACTS_DIR / "config_resolved.json",
            {
                "target_column": target_col,
                "selection_method": target_method,
                "target_disorder_env": str(os.getenv("TARGET_DISORDER", "")),
            },
        )
        feature_report = select_features(dataset_df, questionnaire_df, target_col=target_col)
        y_all, y_mapping = _normalize_target_binary(dataset_df[target_col])
        if y_all.nunique(dropna=True) < 2:
            raise ValueError("El target tiene una sola clase. No se puede entrenar.")

        # Drop rows where target is missing.
        valid_idx = y_all.dropna().index
        dataset_valid = dataset_df.loc[valid_idx].reset_index(drop=True)
        y_all = y_all.loc[valid_idx].astype(int).reset_index(drop=True)

        idx_train, idx_val, idx_test, split_meta = _split_indices(y_all)
        y_train = y_all.iloc[idx_train].reset_index(drop=True)
        y_val = y_all.iloc[idx_val].reset_index(drop=True)
        y_test = y_all.iloc[idx_test].reset_index(drop=True)

        feature_candidates = _build_feature_set_candidates(
            dataset_df=dataset_valid,
            questionnaire_df=questionnaire_df,
            base_features=feature_report["features_used"],
            target_col=target_col,
        )

        param_dist = hyperparameter_space()
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        n_iter_search = 60
        candidate_results: List[Dict[str, Any]] = []
        best_bundle: Dict[str, Any] | None = None

        for candidate in feature_candidates:
            selected_features = candidate["features"]
            X_all = prepare_features_frame(dataset_valid, selected_features)
            X_train = X_all.iloc[idx_train].reset_index(drop=True)
            X_val = X_all.iloc[idx_val].reset_index(drop=True)
            X_test = X_all.iloc[idx_test].reset_index(drop=True)

            preprocessor, preproc_info = build_preprocessor(X_train)

            def make_search(rf_n_jobs: int, search_n_jobs: int) -> RandomizedSearchCV:
                rf = build_random_forest({"n_jobs": rf_n_jobs})
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", rf),
                    ]
                )
                return RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_dist,
                    n_iter=n_iter_search,
                    scoring={
                        "f1": "f1",
                        "recall": "recall",
                        "precision": "precision",
                        "roc_auc": "roc_auc",
                    },
                    refit=False,
                    cv=cv,
                    random_state=42,
                    n_jobs=search_n_jobs,
                    verbose=0,
                    return_train_score=True,
                )

            search_fit_jobs = -1
            search = make_search(rf_n_jobs=search_fit_jobs, search_n_jobs=search_fit_jobs)
            try:
                search.fit(X_train, y_train)
            except PermissionError as parallel_error:
                audit_event(
                    "training_parallel_fallback",
                    {
                        "reason": str(parallel_error),
                        "fallback": "n_jobs=1",
                        "feature_strategy": candidate["name"],
                    },
                )
                search_fit_jobs = 1
                search = make_search(rf_n_jobs=search_fit_jobs, search_n_jobs=search_fit_jobs)
                search.fit(X_train, y_train)

            best_idx, cv_diagnostics = _pick_best_cv_index(search)
            selected_params = search.cv_results_["params"][best_idx]
            selected_cv_f1 = float(search.cv_results_["mean_test_f1"][best_idx])

            best_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", build_random_forest({"n_jobs": search_fit_jobs})),
                ]
            )
            best_pipeline.set_params(**selected_params)
            best_pipeline.fit(X_train, y_train)

            y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]
            threshold_summary = threshold_search(y_val.to_numpy(), y_val_prob)
            val_metrics_final = evaluate_binary_metrics(
                y_val.to_numpy(),
                y_val_prob,
                threshold=threshold_summary.final_threshold,
            )
            val_metrics_05 = evaluate_binary_metrics(y_val.to_numpy(), y_val_prob, threshold=0.5)
            y_train_prob = best_pipeline.predict_proba(X_train)[:, 1]
            train_metrics_final = evaluate_binary_metrics(
                y_train.to_numpy(),
                y_train_prob,
                threshold=threshold_summary.final_threshold,
            )
            train_metrics_05 = evaluate_binary_metrics(
                y_train.to_numpy(),
                y_train_prob,
                threshold=0.5,
            )
            overfit_gap_train_val = max(
                0.0,
                float(train_metrics_final["f1"]) - float(val_metrics_final["f1"]),
            )
            extreme_val_metric_penalty = max(0.0, _max_primary_metric(val_metrics_final) - 0.98)
            model_selection_score = (
                float(val_metrics_final["f1"])
                + 0.18 * float(val_metrics_final["recall"])
                + 0.03 * float(val_metrics_final["precision"])
                - 0.60 * overfit_gap_train_val
                - 0.20 * extreme_val_metric_penalty
            )

            candidate_result = {
                "name": candidate["name"],
                "description": candidate["description"],
                "n_features": len(selected_features),
                "features": selected_features,
                "cv_selected_index": int(best_idx),
                "cv_selected_score_f1": selected_cv_f1,
                "cv_selected_params": selected_params,
                "cv_selection_diagnostics": cv_diagnostics,
                "model_selection_score": float(model_selection_score),
                "overfit_gap_train_val_f1": float(overfit_gap_train_val),
                "extreme_val_metric_penalty": float(extreme_val_metric_penalty),
                "train_metrics_threshold_0_5": train_metrics_05,
                "train_metrics_final": train_metrics_final,
                "validation_metrics_threshold_0_5": val_metrics_05,
                "validation_metrics_final": val_metrics_final,
                "thresholds": {
                    "best_f1": threshold_summary.best_f1_threshold,
                    "best_recall": threshold_summary.best_recall_threshold,
                    "final": threshold_summary.final_threshold,
                },
            }
            candidate_results.append(candidate_result)

            if best_bundle is None or _is_better_candidate(candidate_result, best_bundle["result"]):
                best_bundle = {
                    "result": candidate_result,
                    "pipeline": best_pipeline,
                    "preproc_info": preproc_info,
                    "search": search,
                    "threshold_summary": threshold_summary,
                    "X_test": X_test,
                    "selected_features": selected_features,
                }

        if best_bundle is None:
            raise RuntimeError("No se pudo seleccionar un candidato de entrenamiento.")

        selected_features = best_bundle["selected_features"]
        best_pipeline = best_bundle["pipeline"]
        preproc_info = best_bundle["preproc_info"]
        threshold_summary = best_bundle["threshold_summary"]
        X_test = best_bundle["X_test"]
        chosen_candidate = best_bundle["result"]

        # Additional generalization audit on train+validation (without touching test).
        X_selected_all = prepare_features_frame(dataset_valid, selected_features)
        idx_train_val = np.concatenate([idx_train, idx_val])
        X_train_val = X_selected_all.iloc[idx_train_val].reset_index(drop=True)
        y_train_val = y_all.iloc[idx_train_val].reset_index(drop=True)
        cv_preprocessor, _ = build_preprocessor(X_train_val)
        cv_pipeline = Pipeline(
            steps=[
                ("preprocessor", cv_preprocessor),
                ("model", build_random_forest({"n_jobs": 1})),
            ]
        )
        cv_pipeline.set_params(**chosen_candidate["cv_selected_params"])
        cv_scores = cross_validate(
            cv_pipeline,
            X_train_val,
            y_train_val,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring={
                "f1": "f1",
                "recall": "recall",
                "precision": "precision",
                "roc_auc": "roc_auc",
            },
            return_train_score=True,
            n_jobs=1,
        )
        generalization_cv = {
            "f1_test_mean": float(np.mean(cv_scores["test_f1"])),
            "f1_test_std": float(np.std(cv_scores["test_f1"])),
            "f1_train_mean": float(np.mean(cv_scores["train_f1"])),
            "f1_gap_train_test": float(np.mean(cv_scores["train_f1"]) - np.mean(cv_scores["test_f1"])),
            "recall_test_mean": float(np.mean(cv_scores["test_recall"])),
            "recall_test_std": float(np.std(cv_scores["test_recall"])),
            "precision_test_mean": float(np.mean(cv_scores["test_precision"])),
            "precision_test_std": float(np.std(cv_scores["test_precision"])),
            "roc_auc_test_mean": float(np.mean(cv_scores["test_roc_auc"])),
            "roc_auc_test_std": float(np.std(cv_scores["test_roc_auc"])),
        }
        save_json(ARTIFACTS_DIR / "generalization_cv.json", generalization_cv)

        y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
        metrics_at_05 = evaluate_binary_metrics(y_test.to_numpy(), y_test_prob, threshold=0.5)
        metrics_at_best_f1 = evaluate_binary_metrics(
            y_test.to_numpy(), y_test_prob, threshold=threshold_summary.best_f1_threshold
        )
        metrics_at_best_recall = evaluate_binary_metrics(
            y_test.to_numpy(), y_test_prob, threshold=threshold_summary.best_recall_threshold
        )
        metrics_final = evaluate_binary_metrics(
            y_test.to_numpy(), y_test_prob, threshold=threshold_summary.final_threshold
        )
        generalization_alerts = _build_generalization_alerts(
            train_metrics=chosen_candidate["train_metrics_final"],
            validation_metrics=chosen_candidate["validation_metrics_final"],
            test_metrics=metrics_final,
        )

        # Save model artifacts.
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline.named_steps["model"], MODELS_DIR / "model.joblib")
        save_preprocessor(best_pipeline.named_steps["preprocessor"])

        selected_feature_report = {
            "target_used": target_col,
            "strategy_selected": chosen_candidate["name"],
            "strategy_description": chosen_candidate["description"],
            "features_used": selected_features,
            "features_used_count": len(selected_features),
            "features_excluded": feature_report.get("features_excluded", []),
            "candidate_strategies_tested": [
                {
                    "name": c["name"],
                    "n_features": c["n_features"],
                    "validation_f1_final": c["validation_metrics_final"]["f1"],
                    "validation_recall_final": c["validation_metrics_final"]["recall"],
                }
                for c in candidate_results
            ],
            "feature_origins": feature_report.get("feature_origins", {}),
            "decision": (
                "Se eligió la estrategia con mejor F1 en validación y mayor recall como criterio secundario."
            ),
        }
        save_json(ARTIFACTS_DIR / "selected_features.json", selected_feature_report)

        feature_schema = build_feature_schema(
            selected_features=selected_features,
            feature_to_metadata=feature_to_metadata,
        )
        importance_payload = _extract_feature_importance(best_pipeline, feature_schema)
        _plot_feature_importance(
            importance_payload["feature_importance_aggregated"],
            ARTIFACTS_DIR / "feature_importance.png",
        )

        # Save threshold and metric artifacts.
        threshold_analysis = {
            "best_f1_threshold": threshold_summary.best_f1_threshold,
            "best_recall_threshold": threshold_summary.best_recall_threshold,
            "final_threshold": threshold_summary.final_threshold,
            "rows": threshold_summary.rows,
            "selection_rule": (
                "Se selecciona el mayor recall entre thresholds con F1 >= 95% del F1 máximo en validación."
            ),
        }
        save_json(ARTIFACTS_DIR / "threshold_analysis.json", threshold_analysis)
        save_json(
            ARTIFACTS_DIR / "model_search_report.json",
            {
                "target_column": target_col,
                "n_iter_search": n_iter_search,
                "cv_folds": 4,
                "feature_candidates": candidate_results,
                "selected_strategy": chosen_candidate["name"],
                "selected_validation_metrics": chosen_candidate["validation_metrics_final"],
            },
        )
        _plot_threshold_curve(threshold_summary.rows, ARTIFACTS_DIR / "threshold_curve.png")

        save_json(
            ARTIFACTS_DIR / "classification_report.json",
            metrics_final["classification_report"],
        )
        _plot_confusion_matrix(metrics_final["confusion_matrix"], ARTIFACTS_DIR / "confusion_matrix.png")

        save_json(
            ARTIFACTS_DIR / "feature_importance.json",
            importance_payload,
        )

        metrics_payload = {
            "target_column": target_col,
            "target_mapping": y_mapping,
            "dataset_rows_used": int(len(dataset_valid)),
            "features_count": len(selected_features),
            "features_used": selected_features,
            "split_meta": split_meta,
            "feature_strategy_selected": chosen_candidate["name"],
            "feature_candidates_tested": len(candidate_results),
            "cv_selected_score_f1": chosen_candidate["cv_selected_score_f1"],
            "cv_selected_params": chosen_candidate["cv_selected_params"],
            "validation_metrics_threshold_0_5": chosen_candidate["validation_metrics_threshold_0_5"],
            "validation_metrics_final": chosen_candidate["validation_metrics_final"],
            "train_metrics_threshold_0_5": chosen_candidate["train_metrics_threshold_0_5"],
            "train_metrics_final": chosen_candidate["train_metrics_final"],
            "overfit_gap_train_val_f1": chosen_candidate["overfit_gap_train_val_f1"],
            "model_selection_score": chosen_candidate["model_selection_score"],
            "generalization_cv": generalization_cv,
            "generalization_alerts": generalization_alerts,
            "test_metrics_threshold_0_5": metrics_at_05,
            "test_metrics_threshold_best_f1": metrics_at_best_f1,
            "test_metrics_threshold_best_recall": metrics_at_best_recall,
            "test_metrics_threshold_final": metrics_final,
        }
        save_json(ARTIFACTS_DIR / "metrics.json", metrics_payload)
        save_json(
            ARTIFACTS_DIR / "generalization_diagnostics.json",
            {
                "train_metrics_final": chosen_candidate["train_metrics_final"],
                "validation_metrics_final": chosen_candidate["validation_metrics_final"],
                "test_metrics_final": metrics_final,
                "cross_validation_summary": generalization_cv,
                "overfit_gap_train_val_f1": chosen_candidate["overfit_gap_train_val_f1"],
                "selection_score": chosen_candidate["model_selection_score"],
                "alerts": generalization_alerts,
                "note": "Brecha train-validación cercana a cero indica menor riesgo de sobreentrenamiento, pero no reemplaza validacion externa.",
            },
        )

        metadata = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": "RandomForestClassifier",
            "dataset_file": dataset_profile.get("dataset_file_used"),
            "dataset_hash_sha256": dataset_profile.get("dataset_hash_sha256"),
            "questionnaire_file": questionnaire_profile.get("questionnaire_file_used"),
            "questionnaire_hash_sha256": questionnaire_profile.get("questionnaire_hash_sha256"),
            "target_column": target_col,
            "target_selection_method": target_method,
            "features_used": selected_features,
            "n_features_used": len(selected_features),
            "feature_strategy_selected": chosen_candidate["name"],
            "feature_candidates_tested": len(candidate_results),
            "hyperparameters": chosen_candidate["cv_selected_params"],
            "cv_selection_diagnostics": chosen_candidate["cv_selection_diagnostics"],
            "thresholds": {
                "default": 0.5,
                "best_f1": threshold_summary.best_f1_threshold,
                "best_recall": threshold_summary.best_recall_threshold,
                "final": threshold_summary.final_threshold,
            },
            "metrics_summary": {
                "precision_final": metrics_final["precision"],
                "recall_final": metrics_final["recall"],
                "f1_final": metrics_final["f1"],
                "accuracy_final": metrics_final["accuracy"],
                "roc_auc_final": metrics_final.get("roc_auc"),
                "pr_auc_final": metrics_final.get("pr_auc"),
                "overfit_gap_train_val_f1": chosen_candidate["overfit_gap_train_val_f1"],
                "generalization_cv_f1_mean": generalization_cv["f1_test_mean"],
                "generalization_cv_f1_std": generalization_cv["f1_test_std"],
                "generalization_cv_f1_gap_train_test": generalization_cv["f1_gap_train_test"],
                "generalization_alerts": generalization_alerts,
            },
            "preprocessing_info": preproc_info,
            "contract_summary": {
                "matching_features_count": contract_report.get("matching_features_count"),
                "questionnaire_features_count": contract_report.get("questionnaire_features_count"),
            },
        }
        save_json(MODELS_DIR / "metadata.json", metadata)
        metadata["model_hash_sha256"] = hash_file(MODELS_DIR / "model.joblib")
        save_json(MODELS_DIR / "metadata.json", metadata)

        audit_event(
            "training_completed",
            {
                "target_selected": target_col,
                "dataset_file": dataset_profile.get("dataset_file_used"),
                "dataset_hash_sha256": dataset_profile.get("dataset_hash_sha256"),
                "questionnaire_file": questionnaire_profile.get("questionnaire_file_used"),
                "questionnaire_hash_sha256": questionnaire_profile.get("questionnaire_hash_sha256"),
                "features_used": selected_features,
                "features_excluded": feature_report.get("features_excluded", []),
                "features_used_count": len(selected_features),
                "feature_strategy_selected": chosen_candidate["name"],
                "feature_candidates_tested": len(candidate_results),
                "threshold_final": threshold_summary.final_threshold,
                "threshold_analysis": {
                    "best_f1_threshold": threshold_summary.best_f1_threshold,
                    "best_recall_threshold": threshold_summary.best_recall_threshold,
                    "selection_rule": "best_f1_with_high_recall",
                },
                "metrics_final": metadata["metrics_summary"],
                "generalization_alerts": generalization_alerts,
                "hyperparameters": chosen_candidate["cv_selected_params"],
                "model_hash_sha256": metadata.get("model_hash_sha256"),
            },
        )

        print("Entrenamiento completado.")
        print("Target:", target_col)
        print("Estrategia de features:", chosen_candidate["name"])
        print("Features usadas:", len(selected_features))
        print("F1 final:", round(metrics_final["f1"], 4))
        print("Recall final:", round(metrics_final["recall"], 4))
        print("Precision final:", round(metrics_final["precision"], 4))
        print("Threshold final:", threshold_summary.final_threshold)
        return 0
    except Exception as exc:
        audit_error("training_failed", exc, {})
        print("Error en entrenamiento:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
