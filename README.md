# Sistema local PGC: estimación preliminar conversacional con Random Forest

Aplicación local, gratuita y auditable para estimar si existe un **patrón compatible** con un dominio clínico (por defecto: **conducta**) usando **RandomForestClassifier** y NLP local basado en reglas.

> **Advertencia obligatoria**
> Este resultado no es un diagnóstico médico y debe ser revisado por un profesional calificado.

## Qué hace

- Backend local: `FastAPI` + `Uvicorn`.
- Frontend local: `HTML + CSS + JavaScript` en modo oscuro tipo chat.
- Modelo obligatorio: `RandomForestClassifier` de `scikit-learn`.
- Parser NLP local por reglas (sin APIs externas, sin LLM remoto, sin internet obligatorio).
- Preguntas humanizadas desde `questionnaire_v16_4_visible_questions_excel_utf8.csv`.
- Flujo conversacional completo:
  - preguntas automáticas al abrir,
  - interpretación de respuesta abierta,
  - aclaraciones contextuales por escala,
  - confirmación antes de avanzar,
  - impresión orientativa final,
  - preguntas posteriores sobre el resultado en el mismo chat.

## Target por defecto y filtrado por dominio

- Si **no** se define `TARGET_DISORDER`, el target por defecto es:
  - `target_domain_conduct_final`
- Para target conducta, las preguntas se filtran por dominio de conducta + demográficos permitidos (`age_years`, `sex_assigned_at_birth`).
- Se bloquean preguntas ADHD cuando el target activo es conducta.

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

Genera:
- `artifacts/validation_report.json`
- `artifacts/compliance_report.json`

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

Valores soportados: `conduct`, `adhd`, `anxiety`, `depression`, `elimination` (y equivalentes definidos en `src/config.py`).

## Ejecutar app local

```bash
python app.py
```

o

```bash
uvicorn src.web_app:app --reload
```

Abrir en:

```text
http://127.0.0.1:8000
```

## Flujo conversacional

1. La app carga modelo y preguntas automáticamente.
2. No existe botón “Cargar preguntas”.
3. El usuario responde en lenguaje natural.
4. El parser interpreta según escala (`binary`, `temporal_0_2`, `observation_0_2`, `frequency_0_3`, `impact_0_3`, `numeric_range`, `categorical`).
5. Si hay información parcial, pide solo lo faltante.
6. Solo avanza al confirmar cada respuesta.
7. Al finalizar, genera **impresión psicológica orientativa** (no diagnóstica).
8. Luego permite preguntas sobre el resultado (`result_qa`).
9. El botón de reinicio limpia toda la sesión y comienza desde cero.

## Detalles técnicos en la interfaz

- La sección **Detalles técnicos** inicia colapsada.
- Al desplegarla por primera vez, carga automáticamente métricas, importancia y matriz de confusión.
- No existen botones “Actualizar métricas” ni “Actualizar importancia”.

## Métricas y control de sobreajuste

- Optimización principal: **F1**.
- Prioridad secundaria: **recall**.
- Umbral optimizado por validación (threshold tuning).
- Reporte de brechas train/valid/test y validación cruzada.
- Si métricas de test superan `0.98`, se registra y muestra advertencia de posible señal directa del target o necesidad de validación externa adicional.
- No se capan métricas artificialmente: se reporta el valor real.

## Comandos rápidos

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
- Threshold: `artifacts/threshold_analysis.json`
- Importancia: `artifacts/feature_importance.json`
- Auditoría: `logs/audit.jsonl`

## Limitaciones

- Es herramienta orientativa, no diagnóstico clínico.
- Requiere validación profesional para decisiones reales.
- El parser por reglas puede pedir aclaraciones en respuestas ambiguas.
- Importancia de variables no implica causalidad.
