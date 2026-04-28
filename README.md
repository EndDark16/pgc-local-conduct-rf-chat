# Sistema local PGC: estimación preliminar con Random Forest

Aplicación local, gratuita, auditable y sin servicios externos para estimar si existe un **patrón compatible** con un dominio clínico (por defecto: **conducta**).

> **Advertencia importante**
> Este resultado no es un diagnóstico médico y debe ser revisado por un profesional calificado.

## Características

- Modelo obligatorio: `RandomForestClassifier`.
- Backend: `FastAPI` + `Uvicorn`.
- Frontend: `HTML + CSS + JavaScript` en modo oscuro tipo chat.
- NLP local por reglas (sin APIs externas ni LLMs remotos).
- Priorización de métricas: **F1** principal y **recall** secundaria.
- Ajuste de threshold con validación.
- Auditoría local en `logs/audit.jsonl`.

## Target por defecto y filtrado por dominio

- Si no se define `TARGET_DISORDER`, el target por defecto es:
  - `target_domain_conduct_final`
- Cuando el target es conducta, las preguntas se filtran por dominio de conducta y demográficos permitidos (`age_years`, `sex_assigned_at_birth`).
- El sistema evita mostrar preguntas ADHD si el target activo es conducta.

## Estructura

```text
data/
src/
models/
artifacts/
logs/
tests/
docs/
web/
web/templates/
web/static/
web/static/css/
web/static/js/
web/static/img/
app.py
train.py
predict.py
setup_project.py
validate_project.py
requirements.txt
README.md
```

## CSV requeridos en `data/`

1. Dataset principal:
- `hybrid_no_external_scores_dataset_ready(1).csv`

2. Cuestionario humanizado:
- `questionnaire_v16_4_visible_questions_excel_utf8.csv`

## Instalación

```bash
pip install -r requirements.txt
```

## Preparación y validación

```bash
python setup_project.py
python validate_project.py
```

Genera `artifacts/validation_report.json`.

## Entrenamiento

```bash
python train.py
```

### Selección de trastorno por variable de entorno

Windows PowerShell:

```powershell
$env:TARGET_DISORDER="conduct"; python train.py
```

macOS/Linux:

```bash
TARGET_DISORDER="conduct" python train.py
```

Valores soportados: `conduct`, `adhd`, `anxiety`, `depression`, `elimination` (y equivalentes en español definidos en `src/config.py`).

## Ejecución local

```bash
python app.py
```

o

```bash
uvicorn src.web_app:app --reload
```

Abrir en navegador:

```text
http://127.0.0.1:8000
```

## Flujo conversacional

1. La app carga automáticamente estado del modelo y preguntas.
2. No existe botón “Cargar preguntas”.
3. El asistente pregunta una por una en modo chat.
4. El usuario responde en lenguaje natural.
5. El parser local interpreta según escala (sí/no, temporal, frecuencia, observación, impacto, rango numérico).
6. Si hay ambigüedad, el asistente pide aclaración específica.
7. Solo avanza cuando la respuesta es válida y confirmada.
8. Al final, genera estimación prudente con disclaimer.

## Comandos principales

```bash
python validate_project.py
python train.py
python app.py
pytest
python docs/generar_documentos_word.py
```

## Artefactos clave

- Modelo: `models/model.joblib`
- Preprocesador: `models/preprocessor.joblib`
- Metadatos: `models/metadata.json`
- Esquema de preguntas: `artifacts/feature_schema.json`
- Métricas: `artifacts/metrics.json`
- Thresholds: `artifacts/threshold_analysis.json`
- Importancia: `artifacts/feature_importance.json`
- Auditoría: `logs/audit.jsonl`

## Limitaciones

- Es una herramienta preliminar, no diagnóstica.
- Depende de la calidad del dataset.
- El parser por reglas puede requerir aclaraciones en respuestas ambiguas.
- Importancia de variables no implica causalidad clínica.
