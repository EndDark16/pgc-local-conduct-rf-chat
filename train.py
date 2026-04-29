"""Training pipeline with leakage audit and overfit guard for local Random Forest."""

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
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.audit import audit_error, audit_event
from src.config import (
    ARTIFACTS_DIR,
    MAX_ACCEPTABLE_METRIC,
    MODELS_DIR,
    TARGET_COLUMNS,
    ensure_required_dirs,
    resolve_target_column,
)
from src.data_contract import analyze_data_contract
from src.data_loader import load_main_dataset
from src.feature_selection import select_features
from src.leakage_audit import run_leakage_audit
from src.model import build_random_forest, hyperparameter_space
from src.preprocessing import build_feature_schema, build_preprocessor, prepare_features_frame, save_preprocessor
from src.questionnaire_loader import load_questionnaire
from src.training_utils import (
    evaluate_binary_metrics,
    metrics_above_limit,
    select_model_with_overfit_guard,
    threshold_search,
)
from src.utils import hash_file, save_json


def _safe_stratify_vector(y: pd.Series) -> pd.Series | None:
    counts = y.value_counts(dropna=False)
    if len(counts) < 2:
        return None
    if counts.min() < 2:
        return None
    return y


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


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return series.astype("category").cat.codes.replace(-1, np.nan).astype(float)


def _reduce_high_redundancy_features(
    dataset_df: pd.DataFrame,
    features: List[str],
    y: pd.Series,
    threshold: float = 0.93,
) -> List[str]:
    if len(features) <= 6:
        return sorted(set(features))

    num_df = pd.DataFrame({f: _coerce_numeric(dataset_df[f]) for f in features if f in dataset_df.columns})
    if num_df.shape[1] <= 1:
        return sorted(set(features))

    corr_with_target = {
        col: abs(float(num_df[col].corr(pd.to_numeric(y, errors="coerce")) or 0.0))
        for col in num_df.columns
    }
    corr_matrix = num_df.corr().abs()
    to_drop: set[str] = set()
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            if c1 in to_drop or c2 in to_drop:
                continue
            if float(corr_matrix.loc[c1, c2]) >= threshold:
                drop = c1 if corr_with_target.get(c1, 0.0) >= corr_with_target.get(c2, 0.0) else c2
                to_drop.add(drop)

    reduced = [f for f in features if f not in to_drop]
    return sorted(set(reduced)) if len(reduced) >= 6 else sorted(set(features))


def _build_regularized_space(strict: bool = False) -> Dict[str, List[Any]]:
    if not strict:
        return hyperparameter_space()
    return {
        "model__n_estimators": [100, 150],
        "model__max_depth": [3, 4],
        "model__min_samples_split": [30, 40, 60],
        "model__min_samples_leaf": [10, 15],
        "model__max_features": ["sqrt", "log2", 0.4],
        "model__class_weight": ["balanced", "balanced_subsample"],
        "model__criterion": ["gini", "entropy"],
        "model__bootstrap": [True],
        "model__max_samples": [0.5, 0.6, 0.7],
        "model__ccp_alpha": [0.002, 0.005, 0.01],
    }


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
        high_metric_penalty = max(0.0, mean_test_f1 - MAX_ACCEPTABLE_METRIC)

        selection_score = (
            mean_test_f1
            + 0.17 * mean_test_recall
            + 0.03 * mean_test_precision
            - 0.65 * overfit_gap
            - 0.05 * std_test_f1
            - 0.35 * high_metric_penalty
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
                "high_metric_penalty": high_metric_penalty,
                "selection_score": selection_score,
            }

    return best_idx, diagnostics


def _extract_feature_importance(pipeline: Pipeline, feature_schema: Dict[str, Any]) -> Dict[str, Any]:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
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
    agg_items = []
    for feat, score in aggregated.items():
        meta = metadata_map.get(feat, {})
        label = (
            str(meta.get("feature_label_human") or "").strip()
            or str(meta.get("caregiver_question") or meta.get("question_text_primary") or "").strip()
            or feat
        )
        agg_items.append(
            {
                "feature": feat,
                "importance": float(score),
                "label": label,
            }
        )
    agg_items.sort(key=lambda x: x["importance"], reverse=True)
    total = sum(float(item["importance"]) for item in agg_items) or 1.0
    top_share = float(agg_items[0]["importance"] / total) if agg_items else 0.0

    return {
        "top_transformed_features": raw_items[:80],
        "feature_importance_aggregated": agg_items,
        "importance_concentration_top1": top_share,
    }


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


def _plot_feature_importance(agg_items: list[dict], output_path: Path, top_n: int = 20) -> None:
    top = agg_items[:top_n]
    labels = [str(item["label"])[:70] for item in top][::-1]
    values = [item["importance"] for item in top][::-1]

    plt.figure(figsize=(11, 7))
    plt.barh(labels, values)
    plt.title("Importancia de variables (Random Forest)")
    plt.xlabel("Importancia")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_model_comparison(rows: List[Dict[str, Any]], selected_variant: str, output_path: Path) -> None:
    labels = [str(r["variant_name"]) for r in rows]
    f1_values = [float(r["test_metrics_threshold_final"].get("f1") or 0.0) for r in rows]
    recall_values = [float(r["test_metrics_threshold_final"].get("recall") or 0.0) for r in rows]
    x = np.arange(len(rows))

    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.15, f1_values, width=0.3, label="F1")
    plt.bar(x + 0.15, recall_values, width=0.3, label="Recall")
    plt.axhline(MAX_ACCEPTABLE_METRIC, color="crimson", linestyle="--", linewidth=1.5, label="Límite 0.98")
    for idx, label in enumerate(labels):
        if label == selected_variant:
            plt.scatter(idx, max(f1_values[idx], recall_values[idx]) + 0.01, color="green", s=80, marker="*")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("Comparación de variantes con overfit guard")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _strict_overfit_alert(metrics: Dict[str, Any]) -> str | None:
    above = metrics_above_limit(metrics, MAX_ACCEPTABLE_METRIC)
    if above:
        return (
            "El desempeño sigue siendo extremadamente alto incluso después de controles de regularización. "
            "Esto sugiere que el target puede estar definido de forma muy cercana a las variables usadas. "
            "Se recomienda validación externa con otro dataset."
        )
    return None


def _run_variant_training(
    variant_name: str,
    description: str,
    selected_features: List[str],
    dataset_valid: pd.DataFrame,
    y_all: pd.Series,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    param_dist: Dict[str, List[Any]],
    n_iter_search: int,
) -> Dict[str, Any]:
    X_all = prepare_features_frame(dataset_valid, selected_features)
    X_train = X_all.iloc[idx_train].reset_index(drop=True)
    X_val = X_all.iloc[idx_val].reset_index(drop=True)
    X_test = X_all.iloc[idx_test].reset_index(drop=True)
    y_train = y_all.iloc[idx_train].reset_index(drop=True)
    y_val = y_all.iloc[idx_val].reset_index(drop=True)
    y_test = y_all.iloc[idx_test].reset_index(drop=True)

    preprocessor, preproc_info = build_preprocessor(X_train)
    rf = build_random_forest({"n_jobs": -1})
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf),
        ]
    )
    search = RandomizedSearchCV(
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
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    try:
        search.fit(X_train, y_train)
    except PermissionError:
        search = RandomizedSearchCV(
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
            cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=0,
            return_train_score=True,
        )
        search.fit(X_train, y_train)

    best_idx, cv_diag = _pick_best_cv_index(search)
    selected_params = search.cv_results_["params"][best_idx]

    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_random_forest({"n_jobs": 1})),
        ]
    )
    best_pipeline.set_params(**selected_params)
    best_pipeline.fit(X_train, y_train)

    y_train_prob = best_pipeline.predict_proba(X_train)[:, 1]
    y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]

    threshold_summary = threshold_search(
        y_val.to_numpy(),
        y_val_prob,
        max_acceptable_metric=MAX_ACCEPTABLE_METRIC,
        conservative_recall_ceiling=min(MAX_ACCEPTABLE_METRIC, 0.95),
    )
    train_metrics_05 = evaluate_binary_metrics(y_train.to_numpy(), y_train_prob, threshold=0.5)
    val_metrics_05 = evaluate_binary_metrics(y_val.to_numpy(), y_val_prob, threshold=0.5)
    test_metrics_05 = evaluate_binary_metrics(y_test.to_numpy(), y_test_prob, threshold=0.5)

    train_metrics_final = evaluate_binary_metrics(
        y_train.to_numpy(),
        y_train_prob,
        threshold=threshold_summary.final_threshold,
    )
    val_metrics_final = evaluate_binary_metrics(
        y_val.to_numpy(),
        y_val_prob,
        threshold=threshold_summary.final_threshold,
    )
    test_metrics_final = evaluate_binary_metrics(
        y_test.to_numpy(),
        y_test_prob,
        threshold=threshold_summary.final_threshold,
    )

    idx_train_val = np.concatenate([idx_train, idx_val])
    X_train_val = X_all.iloc[idx_train_val].reset_index(drop=True)
    y_train_val = y_all.iloc[idx_train_val].reset_index(drop=True)
    cv_preprocessor, _ = build_preprocessor(X_train_val)
    cv_pipeline = Pipeline(
        steps=[
            ("preprocessor", cv_preprocessor),
            ("model", build_random_forest({"n_jobs": 1, **selected_params.get("model__", {})})),
        ]
    )
    cv_pipeline.set_params(**selected_params)
    cv_scores = cross_validate(
        cv_pipeline,
        X_train_val,
        y_train_val,
        cv=RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=42),
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
        "precision_test_mean": float(np.mean(cv_scores["test_precision"])),
        "roc_auc_test_mean": float(np.mean(cv_scores["test_roc_auc"])),
    }

    overfit_gap_train_test = max(
        0.0,
        float(train_metrics_final.get("f1") or 0.0) - float(test_metrics_final.get("f1") or 0.0),
    )

    return {
        "variant_name": variant_name,
        "description": description,
        "features": selected_features,
        "n_features": len(selected_features),
        "cv_selected_index": int(best_idx),
        "cv_selected_params": selected_params,
        "cv_selection_diagnostics": cv_diag,
        "thresholds": {
            "best_f1": threshold_summary.best_f1_threshold,
            "best_recall": threshold_summary.best_recall_threshold,
            "final": threshold_summary.final_threshold,
        },
        "threshold_rows": threshold_summary.rows,
        "train_metrics_threshold_0_5": train_metrics_05,
        "validation_metrics_threshold_0_5": val_metrics_05,
        "test_metrics_threshold_0_5": test_metrics_05,
        "train_metrics_final": train_metrics_final,
        "validation_metrics_final": val_metrics_final,
        "test_metrics_threshold_final": test_metrics_final,
        "overfit_gap_train_test_f1": float(overfit_gap_train_test),
        "generalization_cv": generalization_cv,
        "preprocessing_info": preproc_info,
        "_pipeline": best_pipeline,
    }


def main() -> int:
    ensure_required_dirs()
    audit_event("training_started", {"mode": "overfit_guard"})

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

        valid_idx = y_all.dropna().index
        dataset_valid = dataset_df.loc[valid_idx].reset_index(drop=True)
        y_all = y_all.loc[valid_idx].astype(int).reset_index(drop=True)
        idx_train, idx_val, idx_test, split_meta = _split_indices(y_all)

        base_features = [f for f in feature_report.get("features_used", []) if f in dataset_valid.columns]
        leakage_audit = run_leakage_audit(
            dataset_df=dataset_valid,
            y_binary=y_all,
            candidate_features=base_features,
            target_col=target_col,
            persist=True,
        )
        excluded_by_name = {item["feature"] for item in leakage_audit.get("columns_excluded", [])}
        baseline_features = [f for f in base_features if f not in excluded_by_name]
        if len(baseline_features) < 6:
            baseline_features = base_features

        corr_features = {
            item["feature"]
            for item in leakage_audit.get("high_correlation_features", [])
            if float(item.get("abs_correlation") or 0.0) >= 0.92
        }
        suspicious_by_rule = {
            item["feature"]
            for item in leakage_audit.get("suspicious_features", [])
            if "near_deterministic_separation" in item.get("flags", [])
        }

        variant_feature_sets: Dict[str, List[str]] = {
            "baseline_conservador": sorted(set(baseline_features)),
            "sin_alta_correlacion": sorted([f for f in baseline_features if f not in corr_features]) or sorted(set(baseline_features)),
            "reducido_menos_redundante": _reduce_high_redundancy_features(dataset_valid, baseline_features, y_all),
            "sin_deterministicas_regla": sorted([f for f in baseline_features if f not in suspicious_by_rule]) or sorted(set(baseline_features)),
        }

        variant_defs = [
            ("baseline_conservador", "Modelo base conservador con auditoría de nombres."),
            ("sin_alta_correlacion", "Excluye variables con correlación muy alta con el target."),
            ("reducido_menos_redundante", "Reduce redundancia entre variables altamente correlacionadas."),
            ("sin_deterministicas_regla", "Excluye variables con separación casi determinística por valores."),
        ]

        results: List[Dict[str, Any]] = []
        model_bundles: Dict[str, Dict[str, Any]] = {}

        # Train baseline first to derive dominant-feature variant.
        baseline_result = _run_variant_training(
            variant_name="baseline_conservador",
            description="Modelo base conservador con auditoría de nombres.",
            selected_features=variant_feature_sets["baseline_conservador"],
            dataset_valid=dataset_valid,
            y_all=y_all,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            param_dist=_build_regularized_space(strict=False),
            n_iter_search=28,
        )
        baseline_pipeline = baseline_result.pop("_pipeline")
        baseline_schema = build_feature_schema(
            selected_features=baseline_result["features"],
            feature_to_metadata=feature_to_metadata,
            persist=False,
        )
        baseline_importance = _extract_feature_importance(baseline_pipeline, baseline_schema)
        baseline_result["importance_concentration"] = baseline_importance["importance_concentration_top1"]
        baseline_result["top_dominant_feature"] = (
            baseline_importance["feature_importance_aggregated"][0]["feature"]
            if baseline_importance["feature_importance_aggregated"]
            else None
        )
        results.append(baseline_result)
        model_bundles["baseline_conservador"] = {
            "pipeline": baseline_pipeline,
            "importance": baseline_importance,
        }

        dominant_features = [
            item["feature"]
            for item in baseline_importance["feature_importance_aggregated"]
            if float(item.get("importance") or 0.0) >= 0.35
        ]
        if not dominant_features and baseline_importance["feature_importance_aggregated"]:
            top = baseline_importance["feature_importance_aggregated"][0]
            if float(top.get("importance") or 0.0) >= 0.28:
                dominant_features = [top["feature"]]

        no_dominant_features = [f for f in baseline_features if f not in set(dominant_features)]
        if len(no_dominant_features) >= 6:
            variant_feature_sets["sin_importancia_dominante"] = sorted(set(no_dominant_features))
            variant_defs.append(
                (
                    "sin_importancia_dominante",
                    "Elimina variables con importancia excesivamente dominante.",
                )
            )

        # Train remaining variants + strict regularized variant.
        for variant_name, description in variant_defs:
            if variant_name == "baseline_conservador":
                continue
            features = variant_feature_sets.get(variant_name, [])
            if len(features) < 6:
                continue

            result = _run_variant_training(
                variant_name=variant_name,
                description=description,
                selected_features=features,
                dataset_valid=dataset_valid,
                y_all=y_all,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=idx_test,
                param_dist=_build_regularized_space(strict=False),
                n_iter_search=24,
            )
            pipeline = result.pop("_pipeline")
            schema_local = build_feature_schema(selected_features=features, feature_to_metadata=feature_to_metadata, persist=False)
            importance_payload = _extract_feature_importance(pipeline, schema_local)
            result["importance_concentration"] = importance_payload["importance_concentration_top1"]
            results.append(result)
            model_bundles[variant_name] = {"pipeline": pipeline, "importance": importance_payload}

        strict_result = _run_variant_training(
            variant_name="regularizacion_estricta",
            description="Hiperparámetros más restrictivos para control fuerte de sobreajuste.",
            selected_features=variant_feature_sets["baseline_conservador"],
            dataset_valid=dataset_valid,
            y_all=y_all,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            param_dist=_build_regularized_space(strict=True),
            n_iter_search=18,
        )
        strict_pipeline = strict_result.pop("_pipeline")
        strict_schema = build_feature_schema(
            selected_features=strict_result["features"],
            feature_to_metadata=feature_to_metadata,
            persist=False,
        )
        strict_importance = _extract_feature_importance(strict_pipeline, strict_schema)
        strict_result["importance_concentration"] = strict_importance["importance_concentration_top1"]
        results.append(strict_result)
        model_bundles["regularizacion_estricta"] = {"pipeline": strict_pipeline, "importance": strict_importance}

        selected_result, overfit_guard_report = select_model_with_overfit_guard(
            results=results,
            max_acceptable_metric=MAX_ACCEPTABLE_METRIC,
        )
        selected_variant = str(selected_result["variant_name"])
        selected_pipeline = model_bundles[selected_variant]["pipeline"]
        selected_importance = model_bundles[selected_variant]["importance"]

        final_metrics = selected_result["test_metrics_threshold_final"]
        overfit_warning_text = _strict_overfit_alert(final_metrics)
        if overfit_warning_text and not overfit_guard_report.get("overfit_warning"):
            overfit_guard_report["overfit_warning"] = True
        if overfit_warning_text:
            overfit_guard_report["external_validation_note"] = overfit_warning_text

        selected_features = selected_result["features"]
        feature_schema = build_feature_schema(
            selected_features=selected_features,
            feature_to_metadata=feature_to_metadata,
        )

        # Persist model artifacts.
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(selected_pipeline.named_steps["model"], MODELS_DIR / "model.joblib")
        save_preprocessor(selected_pipeline.named_steps["preprocessor"])

        # Persist plots and artifacts.
        _plot_threshold_curve(selected_result["threshold_rows"], ARTIFACTS_DIR / "threshold_curve.png")
        _plot_confusion_matrix(final_metrics["confusion_matrix"], ARTIFACTS_DIR / "confusion_matrix.png")
        _plot_feature_importance(selected_importance["feature_importance_aggregated"], ARTIFACTS_DIR / "feature_importance.png")
        _plot_model_comparison(results, selected_variant, ARTIFACTS_DIR / "model_comparison.png")

        save_json(ARTIFACTS_DIR / "feature_importance.json", selected_importance)
        save_json(ARTIFACTS_DIR / "classification_report.json", final_metrics["classification_report"])

        threshold_analysis = {
            "best_f1_threshold": selected_result["thresholds"]["best_f1"],
            "best_recall_threshold": selected_result["thresholds"]["best_recall"],
            "final_threshold": selected_result["thresholds"]["final"],
            "rows": selected_result["threshold_rows"],
            "selection_rule": (
                "threshold por validacion con overfit guard: maximiza F1 "
                "dentro de limite de metricas y techo conservador de recall"
            ),
        }
        save_json(ARTIFACTS_DIR / "threshold_analysis.json", threshold_analysis)

        comparison_payload = {
            "target_column": target_col,
            "max_acceptable_metric": MAX_ACCEPTABLE_METRIC,
            "variants": [
                {
                    "variant_name": r["variant_name"],
                    "description": r["description"],
                    "n_features": r["n_features"],
                    "features": r["features"],
                    "test_metrics_threshold_final": r["test_metrics_threshold_final"],
                    "train_metrics_final": r["train_metrics_final"],
                    "validation_metrics_final": r["validation_metrics_final"],
                    "importance_concentration": r.get("importance_concentration"),
                    "metrics_above_limit": metrics_above_limit(r["test_metrics_threshold_final"], MAX_ACCEPTABLE_METRIC),
                    "overfit_gap_train_test_f1": r.get("overfit_gap_train_test_f1"),
                }
                for r in results
            ],
            "selected_variant": selected_variant,
        }
        save_json(ARTIFACTS_DIR / "model_comparison.json", comparison_payload)
        save_json(ARTIFACTS_DIR / "overfit_guard_report.json", overfit_guard_report)

        selected_feature_report = {
            "target_used": target_col,
            "strategy_selected": selected_variant,
            "features_used": selected_features,
            "features_used_count": len(selected_features),
            "features_excluded": feature_report.get("features_excluded", []),
            "features_excluded_by_leakage_name": leakage_audit.get("columns_excluded", []),
            "warnings": [
                overfit_warning_text
                if overfit_warning_text
                else "No se detectó advertencia crítica de sobreajuste por límite máximo.",
            ],
            "feature_origins": feature_report.get("feature_origins", {}),
        }
        save_json(ARTIFACTS_DIR / "selected_features.json", selected_feature_report)

        metrics_payload = {
            "target_column": target_col,
            "target_mapping": y_mapping,
            "dataset_rows_used": int(len(dataset_valid)),
            "features_count": len(selected_features),
            "features_used": selected_features,
            "split_meta": split_meta,
            "model_variant_selected": selected_variant,
            "overfit_guard_applied": True,
            "max_acceptable_metric": MAX_ACCEPTABLE_METRIC,
            "metrics_above_limit": metrics_above_limit(final_metrics, MAX_ACCEPTABLE_METRIC),
            "overfit_warning": bool(overfit_guard_report.get("overfit_warning", False)),
            "overfit_warning_text": overfit_warning_text,
            "selected_model_reason": overfit_guard_report.get("selected_model_reason"),
            "test_metrics_threshold_0_5": selected_result["test_metrics_threshold_0_5"],
            "test_metrics_threshold_final": final_metrics,
            "train_metrics_final": selected_result["train_metrics_final"],
            "validation_metrics_final": selected_result["validation_metrics_final"],
            "generalization_cv": selected_result["generalization_cv"],
            "thresholds": selected_result["thresholds"],
            "leakage_audit_summary": leakage_audit.get("summary", {}),
        }
        save_json(ARTIFACTS_DIR / "metrics.json", metrics_payload)

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
            "model_variant_selected": selected_variant,
            "max_acceptable_metric": MAX_ACCEPTABLE_METRIC,
            "overfit_guard_applied": True,
            "overfit_warning": bool(overfit_guard_report.get("overfit_warning", False)),
            "metrics_above_limit": metrics_above_limit(final_metrics, MAX_ACCEPTABLE_METRIC),
            "selected_model_reason": overfit_guard_report.get("selected_model_reason"),
            "leakage_audit_summary": leakage_audit.get("summary", {}),
            "hyperparameters": selected_result["cv_selected_params"],
            "thresholds": {
                "default": 0.5,
                "best_f1": selected_result["thresholds"]["best_f1"],
                "best_recall": selected_result["thresholds"]["best_recall"],
                "final": selected_result["thresholds"]["final"],
            },
            "metrics_summary": {
                "precision_final": final_metrics.get("precision"),
                "recall_final": final_metrics.get("recall"),
                "f1_final": final_metrics.get("f1"),
                "accuracy_final": final_metrics.get("accuracy"),
                "roc_auc_final": final_metrics.get("roc_auc"),
                "pr_auc_final": final_metrics.get("pr_auc"),
                "overfit_gap_train_test_f1": selected_result.get("overfit_gap_train_test_f1"),
                "generalization_cv_f1_mean": selected_result.get("generalization_cv", {}).get("f1_test_mean"),
                "generalization_cv_f1_std": selected_result.get("generalization_cv", {}).get("f1_test_std"),
            },
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
                "model_variant_selected": selected_variant,
                "features_used_count": len(selected_features),
                "threshold_final": selected_result["thresholds"]["final"],
                "metrics_final": metadata["metrics_summary"],
                "metrics_above_limit": metadata["metrics_above_limit"],
                "overfit_warning": metadata["overfit_warning"],
                "selected_model_reason": metadata["selected_model_reason"],
                "leakage_audit_summary": metadata["leakage_audit_summary"],
                "model_hash_sha256": metadata.get("model_hash_sha256"),
            },
        )

        print("Entrenamiento completado.")
        print("Target:", target_col)
        print("Variante seleccionada:", selected_variant)
        print("Features usadas:", len(selected_features))
        print("F1 final:", round(float(final_metrics["f1"]), 4))
        print("Recall final:", round(float(final_metrics["recall"]), 4))
        print("Precision final:", round(float(final_metrics["precision"]), 4))
        print("Threshold final:", selected_result["thresholds"]["final"])
        if overfit_warning_text:
            print("ADVERTENCIA:", overfit_warning_text)
        return 0
    except Exception as exc:
        audit_error("training_failed", exc, {})
        print("Error en entrenamiento:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
