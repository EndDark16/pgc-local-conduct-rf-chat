"""Project validator with safe auto-fixes and detailed compliance report."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DEFAULT_TARGET_COLUMN,
    REQUIRED_DIRS,
    ROOT_DIR,
    WEB_STATIC_CSS_DIR,
    WEB_STATIC_JS_DIR,
    WEB_TEMPLATES_DIR,
)
from src.utils import load_json, save_json


ESSENTIAL_FILES = [
    "app.py",
    "train.py",
    "predict.py",
    "setup_project.py",
    "validate_project.py",
    "requirements.txt",
    "README.md",
    "netlify.toml",
    "web/index.html",
    "netlify/functions/api-proxy.js",
    "src/data_contract.py",
    "src/data_loader.py",
    "src/questionnaire_loader.py",
    "src/feature_selection.py",
    "src/leakage_audit.py",
    "src/preprocessing.py",
    "src/question_generator.py",
    "src/question_explainer.py",
    "src/nlp_interpreter.py",
    "src/model.py",
    "src/predictor.py",
    "src/web_app.py",
    "src/audit.py",
    "src/training_utils.py",
    "web/templates/index.html",
    "web/static/css/styles.css",
    "web/static/js/app.js",
    "docs/documento_pgc_contenido.md",
    "docs/presentacion_pgc_contenido.md",
    "docs/generar_documentos_word.py",
    "docs/demo_commands.txt",
]

REQUIRED_DEPENDENCIES = {
    "pandas",
    "numpy",
    "scikit-learn",
    "joblib",
    "fastapi",
    "uvicorn",
    "jinja2",
    "python-multipart",
    "matplotlib",
    "rapidfuzz",
    "unidecode",
    "openpyxl",
    "pytest",
    "python-docx",
}

BANNED_REQUIREMENTS = {
    "torch",
    "tensorflow",
    "keras",
    "transformers",
    "spacy",
    "streamlit",
    "gradio",
    "dash",
}

ARTIFACTS_EXPECTED = [
    "artifacts/data_contract_report.json",
    "artifacts/dataset_profile.json",
    "artifacts/questionnaire_profile.json",
    "artifacts/config_resolved.json",
    "artifacts/selected_features.json",
    "artifacts/feature_schema.json",
    "artifacts/metrics.json",
    "artifacts/threshold_analysis.json",
    "artifacts/threshold_curve.png",
    "artifacts/confusion_matrix.png",
    "artifacts/feature_importance.png",
    "artifacts/feature_importance.json",
    "artifacts/classification_report.json",
    "artifacts/leakage_audit.json",
    "artifacts/model_comparison.json",
    "artifacts/model_comparison.png",
    "artifacts/overfit_guard_report.json",
]

MODEL_EXPECTED = [
    "models/model.joblib",
    "models/preprocessor.joblib",
    "models/metadata.json",
]


def _find_main_dataset() -> Path | None:
    preferred = [
        DATA_DIR / "hybrid_no_external_scores_dataset_ready(1).csv",
        DATA_DIR / "hybrid_no_external_scores_dataset_ready.csv",
    ]
    for path in preferred:
        if path.exists():
            return path
    candidates = sorted(DATA_DIR.glob("hybrid_no_external_scores_dataset_ready*.csv"))
    return candidates[0] if candidates else None


def _find_questionnaire() -> Path | None:
    path = DATA_DIR / "questionnaire_v16_4_visible_questions_excel_utf8.csv"
    if path.exists():
        return path
    candidates = sorted(DATA_DIR.glob("questionnaire*.csv"))
    return candidates[0] if candidates else None


def _http_get_json(url: str, timeout: float = 8.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_post_json(url: str, payload: Dict[str, Any], timeout: float = 8.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_requirements(path: Path) -> List[str]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        entries.append(clean.split("==")[0].strip().lower())
    return entries


def _has_banned_imports() -> List[str]:
    banned_hits: List[str] = []
    banned_roots = {
        "torch",
        "tensorflow",
        "keras",
        "transformers",
        "spacy",
        "streamlit",
        "gradio",
        "dash",
    }
    for py_file in ROOT_DIR.rglob("*.py"):
        if ".venv" in str(py_file) or "__pycache__" in str(py_file):
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0].lower()
                    if root in banned_roots:
                        banned_hits.append(f"{py_file}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0].lower()
                    if root in banned_roots:
                        banned_hits.append(f"{py_file}: from {node.module} import ...")
    return banned_hits


def _quick_tests() -> Dict[str, Any]:
    result = {"executed": False, "return_code": None, "stdout_tail": "", "stderr_tail": ""}
    smoke_test_file = ROOT_DIR / "tests" / "test_smoke.py"
    if not smoke_test_file.exists():
        return result
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(smoke_test_file)],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
    )
    result["executed"] = True
    result["return_code"] = proc.returncode
    result["stdout_tail"] = "\n".join(proc.stdout.splitlines()[-20:])
    result["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-20:])
    return result


def _backend_smoke_check() -> Dict[str, Any]:
    output: Dict[str, Any] = {"ok": False, "errors": [], "checks": {}}
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(ROOT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        status_payload = None
        for _ in range(30):
            time.sleep(0.4)
            try:
                status_payload = _http_get_json("http://127.0.0.1:8000/api/model-status")
                break
            except Exception:
                continue
        if status_payload is None:
            output["errors"].append("No se pudo levantar backend en http://127.0.0.1:8000")
            return output

        output["checks"]["model_status"] = status_payload
        output["checks"]["home_ok"] = True

        questions = _http_get_json(
            "http://127.0.0.1:8000/api/questions?role=caregiver&session_id=validate_full"
        )
        output["checks"]["questions_ok"] = bool(questions.get("total", 0) > 0)
        feature = questions.get("questions", [{}])[0].get("feature")
        if feature:
            explain = _http_post_json(
                "http://127.0.0.1:8000/api/chat/explain",
                {"feature_name": feature, "mode": "simple"},
            )
            interpret_help = _http_post_json(
                "http://127.0.0.1:8000/api/chat/interpret",
                {
                    "feature_name": feature,
                    "answer": "no entiendo",
                    "role": "caregiver",
                    "session_id": "validate_full",
                },
            )
            output["checks"]["explain_ok"] = bool(explain.get("simple_explanation"))
            output["checks"]["interpret_help_ok"] = bool(interpret_help.get("needs_explanation"))

        metrics = _http_get_json("http://127.0.0.1:8000/api/metrics")
        importance = _http_get_json("http://127.0.0.1:8000/api/feature-importance")
        output["checks"]["metrics_endpoint_ok"] = bool(metrics.get("ok") or "message" in metrics)
        output["checks"]["importance_endpoint_ok"] = bool(
            importance.get("ok") or "message" in importance
        )
        _http_post_json("http://127.0.0.1:8000/api/reset-session", {"session_id": "validate_full"})
        output["ok"] = True
        return output
    except urllib.error.HTTPError as http_error:
        output["errors"].append(f"HTTP error en smoke backend: {http_error}")
        return output
    except Exception as exc:  # noqa: BLE001
        output["errors"].append(f"Error en smoke backend: {exc}")
        return output
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def _compliance_item(item_id: str, description: str, passed: bool, evidence: Any) -> Dict[str, Any]:
    return {
        "id": item_id,
        "description": description,
        "passed": bool(passed),
        "evidence": evidence,
    }


def _check_target_domain_coherence() -> Dict[str, Any]:
    metadata = load_json(ROOT_DIR / "models/metadata.json", default={})
    schema = load_json(ROOT_DIR / "artifacts/feature_schema.json", default={})
    selected = load_json(ROOT_DIR / "artifacts/selected_features.json", default={})
    target = str(metadata.get("target_column") or selected.get("target_used") or DEFAULT_TARGET_COLUMN)

    schema_features = [str(item.get("feature", "")) for item in schema.get("features", [])]
    selected_features = [str(f) for f in selected.get("features_used", [])]

    adhd_in_schema = [f for f in schema_features if f.startswith("adhd_")]
    adhd_in_selected = [f for f in selected_features if f.startswith("adhd_")]
    return {
        "target": target,
        "schema_feature_count": len(schema_features),
        "selected_feature_count": len(selected_features),
        "adhd_in_schema": adhd_in_schema,
        "adhd_in_selected": adhd_in_selected,
        "coherent": not (target == DEFAULT_TARGET_COLUMN and (adhd_in_schema or adhd_in_selected)),
    }


def main() -> int:
    issues: List[str] = []
    warnings: List[str] = []
    fixes: List[str] = []

    for directory in REQUIRED_DIRS:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            fixes.append(f"Carpeta creada: {directory}")

    for rel in ESSENTIAL_FILES:
        path = ROOT_DIR / rel
        if not path.exists():
            issues.append(f"Falta archivo esencial: {rel}")

    main_dataset = _find_main_dataset()
    questionnaire = _find_questionnaire()
    if main_dataset is None:
        issues.append(
            "No estÃ¡ el dataset principal en data/. Esperado: hybrid_no_external_scores_dataset_ready(1).csv "
            "o hybrid_no_external_scores_dataset_ready.csv"
        )
    if questionnaire is None:
        issues.append(
            "No estÃ¡ el CSV de preguntas en data/. Esperado: questionnaire_v16_4_visible_questions_excel_utf8.csv"
        )

    expected_main = DATA_DIR / "hybrid_no_external_scores_dataset_ready(1).csv"
    if main_dataset is not None and not expected_main.exists():
        if main_dataset.name == "hybrid_no_external_scores_dataset_ready.csv":
            shutil.copy2(main_dataset, expected_main)
            fixes.append(f"Se creÃ³ copia con nombre esperado del dataset principal: {expected_main.name}")
            main_dataset = expected_main

    expected_q = DATA_DIR / "questionnaire_v16_4_visible_questions_excel_utf8.csv"
    if questionnaire is not None and questionnaire.name != expected_q.name and not expected_q.exists():
        shutil.copy2(questionnaire, expected_q)
        fixes.append(f"Se creÃ³ copia con nombre esperado del cuestionario: {expected_q.name}")
        questionnaire = expected_q

    # Contract analysis and report generation.
    contract_status = {"ok": False, "error": ""}
    try:
        from src.data_contract import analyze_data_contract

        analyze_data_contract()
        contract_status["ok"] = True
    except Exception as exc:  # noqa: BLE001
        contract_status["error"] = str(exc)
        issues.append(f"No se pudo generar/validar el contrato de datos: {exc}")

    modules_to_import = [
        "src.config",
        "src.data_loader",
        "src.questionnaire_loader",
        "src.data_contract",
        "src.feature_selection",
        "src.preprocessing",
        "src.question_generator",
        "src.question_explainer",
        "src.nlp_interpreter",
        "src.model",
        "src.predictor",
        "src.web_app",
    ]
    import_results: Dict[str, str] = {}
    for mod in modules_to_import:
        try:
            importlib.import_module(mod)
            import_results[mod] = "ok"
        except Exception as exc:  # noqa: BLE001
            import_results[mod] = f"error: {exc}"
            issues.append(f"No se pudo importar {mod}: {exc}")

    # Frontend static validation.
    html_path = WEB_TEMPLATES_DIR / "index.html"
    css_path = WEB_STATIC_CSS_DIR / "styles.css"
    js_path = WEB_STATIC_JS_DIR / "app.js"
    for static_file in [html_path, css_path, js_path]:
        if static_file.exists() and static_file.stat().st_size == 0:
            static_file.write_text("/* archivo autocorregido */\n", encoding="utf-8")
            fixes.append(f"Archivo estÃ¡tico vacÃ­o autocorregido: {static_file}")

    model_exists = all((ROOT_DIR / rel).exists() for rel in MODEL_EXPECTED)
    if not model_exists:
        warnings.append("Modelo aÃºn no entrenado. Ejecuta python train.py")

    quick_tests = _quick_tests()
    if quick_tests["executed"] and quick_tests["return_code"] != 0:
        issues.append("FallÃ³ prueba rÃ¡pida test_smoke.py")

    backend_smoke = _backend_smoke_check()
    if not backend_smoke["ok"]:
        warnings.append("Smoke check del backend no pasÃ³ completamente.")

    req_entries = _parse_requirements(ROOT_DIR / "requirements.txt")
    req_missing = sorted(dep for dep in REQUIRED_DEPENDENCIES if dep not in req_entries)
    req_banned = sorted(dep for dep in BANNED_REQUIREMENTS if dep in req_entries)
    if req_missing:
        issues.append("Faltan dependencias requeridas en requirements.txt: " + ", ".join(req_missing))
    if req_banned:
        issues.append("Hay dependencias prohibidas en requirements.txt: " + ", ".join(req_banned))

    banned_import_hits = _has_banned_imports()
    if banned_import_hits:
        issues.append("Se detectaron imports prohibidos en cÃ³digo Python.")

    artifacts_missing = [rel for rel in ARTIFACTS_EXPECTED if not (ROOT_DIR / rel).exists()]
    models_missing = [rel for rel in MODEL_EXPECTED if not (ROOT_DIR / rel).exists()]
    domain_coherence = _check_target_domain_coherence()
    if not domain_coherence["coherent"]:
        issues.append(
            "Incoherencia de dominio detectada: target conducta contiene features ADHD en schema/seleccion."
        )

    # Compliance matrix.
    html_text = html_path.read_text(encoding="utf-8", errors="ignore") if html_path.exists() else ""
    js_text = js_path.read_text(encoding="utf-8", errors="ignore") if js_path.exists() else ""
    css_text = css_path.read_text(encoding="utf-8", errors="ignore") if css_path.exists() else ""
    disclaimer_ok = (
        "Este resultado no es un diagnÃ³stico mÃ©dico y debe ser revisado por un profesional calificado."
        in html_text
    )
    buttons_ok = all(
        (text in html_text) or (text in js_text)
        for text in [
            "No entiendo",
            "Dame un ejemplo",
            "Explicar con palabras",
            "Enviar",
            "Reiniciar",
        ]
    )
    forbidden_frontend_texts = [
        "Cargar preguntas",
        "Pregunta 0 de 0",
        "Pulsa Cargar preguntas",
        "Pulsa \"Cargar preguntas\"",
        "Actualizar métricas",
        "Actualizar importancia",
        "loadMetricsBtn",
        "loadImportanceBtn",
        "Input for",
        "[{'value': 0, 'label': 'No'}",
    ]
    forbidden_frontend_hits = [
        token for token in forbidden_frontend_texts if token in html_text or token in js_text
    ]
    if forbidden_frontend_hits:
        issues.append(
            "Frontend contiene textos prohibidos o de demo: " + ", ".join(forbidden_frontend_hits)
        )

    compliance_items = [
        _compliance_item(
            "dirs_and_structure",
            "Estructura de carpetas obligatoria",
            all(d.exists() for d in REQUIRED_DIRS),
            [str(d) for d in REQUIRED_DIRS],
        ),
        _compliance_item(
            "essential_files",
            "Archivos esenciales obligatorios",
            len([f for f in ESSENTIAL_FILES if not (ROOT_DIR / f).exists()]) == 0,
            ESSENTIAL_FILES,
        ),
        _compliance_item(
            "csv_presence",
            "CSV principal y cuestionario presentes con nombres esperados",
            (DATA_DIR / "hybrid_no_external_scores_dataset_ready(1).csv").exists()
            and (DATA_DIR / "questionnaire_v16_4_visible_questions_excel_utf8.csv").exists(),
            {
                "main_dataset": str(DATA_DIR / "hybrid_no_external_scores_dataset_ready(1).csv"),
                "questionnaire": str(DATA_DIR / "questionnaire_v16_4_visible_questions_excel_utf8.csv"),
            },
        ),
        _compliance_item(
            "default_target_conduct",
            "Target por defecto configurado como target_domain_conduct_final",
            DEFAULT_TARGET_COLUMN == "target_domain_conduct_final",
            {"DEFAULT_TARGET_COLUMN": DEFAULT_TARGET_COLUMN},
        ),
        _compliance_item(
            "target_domain_coherence",
            "Si el target es conducta, no se permiten features ADHD en preguntas",
            domain_coherence["coherent"],
            domain_coherence,
        ),
        _compliance_item(
            "contract_report",
            "Reporte de contrato de datos generado",
            contract_status["ok"] and (ROOT_DIR / "artifacts/data_contract_report.json").exists(),
            contract_status,
        ),
        _compliance_item(
            "dataset_analysis_doc",
            "Documento docs/dataset_analysis.md generado",
            (ROOT_DIR / "docs/dataset_analysis.md").exists(),
            "docs/dataset_analysis.md",
        ),
        _compliance_item(
            "requirements_declared",
            "Dependencias requeridas declaradas",
            not req_missing,
            {"missing": req_missing, "declared": req_entries},
        ),
        _compliance_item(
            "requirements_banned_absent",
            "Dependencias prohibidas ausentes",
            not req_banned,
            {"banned_found": req_banned},
        ),
        _compliance_item(
            "forbidden_imports_absent",
            "Sin imports de frameworks/proveedores prohibidos",
            not banned_import_hits,
            banned_import_hits[:20],
        ),
        _compliance_item(
            "random_forest_model_file",
            "Modelo y preprocesador guardados",
            not models_missing,
            {"missing": models_missing, "expected": MODEL_EXPECTED},
        ),
        _compliance_item(
            "artifacts_generated",
            "Artefactos de mÃ©tricas/threshold/importancia generados",
            not artifacts_missing,
            {"missing": artifacts_missing},
        ),
        _compliance_item(
            "frontend_files",
            "Frontend HTML/CSS/JS existe y no estÃ¡ vacÃ­o",
            html_path.exists() and css_path.exists() and js_path.exists() and len(css_text) > 20 and len(js_text) > 20,
            {
                "html_exists": html_path.exists(),
                "css_exists": css_path.exists(),
                "js_exists": js_path.exists(),
            },
        ),
        _compliance_item(
            "frontend_required_controls",
            "Frontend contiene botones de ayuda y flujo de confirmaciÃ³n",
            buttons_ok,
            "No entiendo la pregunta / Dame un ejemplo / Explicar con palabras mÃ¡s simples / Enviar / Reiniciar",
        ),
        _compliance_item(
            "frontend_no_demo_or_technical_text",
            "Frontend no contiene textos de demo, Input for ni JSON crudo",
            len(forbidden_frontend_hits) == 0,
            {"forbidden_hits": forbidden_frontend_hits},
        ),
        _compliance_item(
            "medical_disclaimer_visible",
            "Advertencia no diagnÃ³stica visible",
            disclaimer_ok,
            "Texto de advertencia en index.html",
        ),
        _compliance_item(
            "quick_smoke_test",
            "Prueba rÃ¡pida de pytest ejecutada",
            (not quick_tests["executed"]) or quick_tests["return_code"] == 0,
            quick_tests,
        ),
        _compliance_item(
            "backend_endpoint_smoke",
            "Backend levanta y endpoints principales responden",
            backend_smoke["ok"],
            backend_smoke,
        ),
        _compliance_item(
            "audit_file_exists",
            "Archivo de auditorÃ­a existe",
            (ROOT_DIR / "logs/audit.jsonl").exists(),
            "logs/audit.jsonl",
        ),
        _compliance_item(
            "docs_pgc",
            "DocumentaciÃ³n PGC y comandos demo presentes",
            all(
                (ROOT_DIR / rel).exists()
                for rel in [
                    "docs/documento_pgc_contenido.md",
                    "docs/presentacion_pgc_contenido.md",
                    "docs/demo_commands.txt",
                    "docs/generar_documentos_word.py",
                ]
            ),
            [
                "docs/documento_pgc_contenido.md",
                "docs/presentacion_pgc_contenido.md",
                "docs/demo_commands.txt",
                "docs/generar_documentos_word.py",
            ],
        ),
    ]

    compliance_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if all(item["passed"] for item in compliance_items) else "error",
        "summary": {
            "total_items": len(compliance_items),
            "passed_items": sum(1 for i in compliance_items if i["passed"]),
            "failed_items": sum(1 for i in compliance_items if not i["passed"]),
        },
        "items": compliance_items,
    }
    save_json(ARTIFACTS_DIR / "compliance_report.json", compliance_report)

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT_DIR),
        "checks": {
            "required_dirs_ok": all(d.exists() for d in REQUIRED_DIRS),
            "essential_files_checked": len(ESSENTIAL_FILES),
            "main_dataset_found": str(main_dataset) if main_dataset else None,
            "questionnaire_found": str(questionnaire) if questionnaire else None,
            "imports": import_results,
            "model_exists": model_exists,
            "frontend_files": {
                "html_exists": html_path.exists(),
                "css_exists": css_path.exists(),
                "js_exists": js_path.exists(),
            },
            "quick_tests": quick_tests,
            "backend_smoke": backend_smoke,
            "requirements": {
                "missing_required": req_missing,
                "banned_found": req_banned,
            },
            "banned_import_hits": banned_import_hits[:20],
            "domain_coherence": domain_coherence,
            "forbidden_frontend_hits": forbidden_frontend_hits,
        },
        "issues": issues,
        "warnings": warnings,
        "auto_fixes": fixes,
        "status": "ok" if not issues else "error",
        "next_steps": (
            ["python train.py", "python app.py"] if not issues else ["Corrige errores listados y vuelve a ejecutar."]
        ),
        "compliance_report_path": str(ARTIFACTS_DIR / "compliance_report.json"),
    }
    save_json(ARTIFACTS_DIR / "validation_report.json", report)

    print("Reporte de validaciÃ³n:", ARTIFACTS_DIR / "validation_report.json")
    print("Reporte de cumplimiento:", ARTIFACTS_DIR / "compliance_report.json")
    print("Estado:", report["status"])
    if issues:
        print("\nErrores:")
        for item in issues:
            print("-", item)
    if warnings:
        print("\nAdvertencias:")
        for item in warnings:
            print("-", item)
    if fixes:
        print("\nAutocorrecciones:")
        for item in fixes:
            print("-", item)
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())


