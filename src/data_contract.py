"""Dataset-questionnaire contract validation and reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import (
    ARTIFACTS_DIR,
    CONDUCT_SECTION_NAME,
    DEFAULT_TARGET_COLUMN,
    DOCS_DIR,
    TARGET_COLUMNS,
    resolve_target_column,
)
from .data_loader import load_main_dataset
from .questionnaire_loader import load_questionnaire
from .feature_selection import is_feature_allowed_for_target
from .utils import normalize_text, save_json


def _detect_conduct_candidates(questionnaire_df) -> List[str]:
    candidates: List[str] = []
    for _, row in questionnaire_df.iterrows():
        feature = row.get("feature")
        if not feature:
            continue
        allowed, _ = is_feature_allowed_for_target(
            str(feature),
            row.to_dict(),
            DEFAULT_TARGET_COLUMN,
        )
        section = normalize_text(row.get("section_name", ""))
        if allowed or section == normalize_text(CONDUCT_SECTION_NAME):
            candidates.append(str(feature))
    return sorted(set(candidates))


def analyze_data_contract() -> Dict[str, Any]:
    dataset_df, dataset_profile = load_main_dataset()
    questionnaire_df, questionnaire_profile, _ = load_questionnaire()

    dataset_columns = set(dataset_df.columns)
    q_features = set(questionnaire_df["feature"].dropna().astype(str))
    matched_features = sorted(q_features.intersection(dataset_columns))
    questionnaire_not_in_dataset = sorted(q_features - dataset_columns)
    dataset_not_in_questionnaire = sorted(
        c for c in dataset_columns - q_features if not c.startswith("target_domain_")
    )

    detected_targets = [c for c in dataset_df.columns if c in TARGET_COLUMNS]
    target_selected, target_method = resolve_target_column(available_targets=detected_targets)
    target_distribution = (
        dataset_df[target_selected].value_counts(dropna=False, normalize=False).to_dict()
        if target_selected in dataset_df.columns
        else {}
    )

    conduct_candidates = _detect_conduct_candidates(questionnaire_df)
    warnings: List[str] = []
    if questionnaire_not_in_dataset:
        warnings.append(
            f"Hay {len(questionnaire_not_in_dataset)} features del cuestionario que no están en el dataset."
        )
    if target_selected != DEFAULT_TARGET_COLUMN:
        warnings.append(
            f"El target seleccionado actual no es el de conducta por defecto: {target_selected}."
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_file": dataset_profile.get("dataset_file_used"),
        "questionnaire_file": questionnaire_profile.get("questionnaire_file_used"),
        "dataset_shape": [int(dataset_df.shape[0]), int(dataset_df.shape[1])],
        "questionnaire_shape": questionnaire_profile.get("source_shape")
        or [int(questionnaire_df.shape[0]), int(questionnaire_df.shape[1])],
        "questionnaire_processed_shape": [int(questionnaire_df.shape[0]), int(questionnaire_df.shape[1])],
        "dataset_main_columns_sample": dataset_df.columns.tolist()[:40],
        "questionnaire_features_count": len(q_features),
        "matching_features_count": len(matched_features),
        "questionnaire_features_not_in_dataset": questionnaire_not_in_dataset,
        "dataset_columns_not_in_questionnaire": dataset_not_in_questionnaire,
        "detected_target_columns": detected_targets,
        "selected_target": target_selected,
        "target_selection_method": target_method,
        "target_distribution_counts": target_distribution,
        "conduct_candidate_features": conduct_candidates,
        "warnings": warnings,
        "decisions": [
            "El cuestionario se usa como contrato oficial de entradas.",
            "Se prioriza target_domain_conduct_final si está disponible.",
            "Las preguntas humanizadas se toman del CSV del cuestionario.",
        ],
    }
    save_json(ARTIFACTS_DIR / "data_contract_report.json", report)
    _write_dataset_analysis_markdown(report)
    return report


def _write_dataset_analysis_markdown(report: Dict[str, Any]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / "dataset_analysis.md"
    text = f"""# Análisis de Contrato de Datos

## Archivos analizados
- Dataset principal: `{report.get("dataset_file")}`
- Cuestionario humanizado: `{report.get("questionnaire_file")}`

## Dimensiones
- Dataset principal: `{report.get("dataset_shape")}`
- Cuestionario: `{report.get("questionnaire_shape")}`

## Relación input-pregunta
- Features del cuestionario: `{report.get("questionnaire_features_count")}`
- Features coincidentes con dataset: `{report.get("matching_features_count")}`
- Features del cuestionario no presentes en dataset: `{len(report.get("questionnaire_features_not_in_dataset", []))}`

## Targets detectados
- Targets: `{report.get("detected_target_columns")}`
- Target seleccionado: `{report.get("selected_target")}`
- Método de selección: `{report.get("target_selection_method")}`

## Distribución del target seleccionado
`{report.get("target_distribution_counts")}`

## Features candidatas de conducta
Total: `{len(report.get("conduct_candidate_features", []))}`

## Advertencias
{chr(10).join(f"- {w}" for w in report.get("warnings", [])) or "- Sin advertencias críticas."}

## Decisiones tomadas
{chr(10).join(f"- {d}" for d in report.get("decisions", []))}
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    result = analyze_data_contract()
    print("Reporte generado:", ARTIFACTS_DIR / "data_contract_report.json")
    print("Target seleccionado:", result.get("selected_target"))
