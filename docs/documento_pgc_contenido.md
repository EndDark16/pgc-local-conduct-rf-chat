# 1. Datos generales

- **Título:** Sistema local de apoyo preliminar para estimar patrones compatibles con trastorno de conducta en niños mediante Random Forest e interpretación local de lenguaje natural
- **Integrantes:** [Nombre de los integrantes]
- **Programa:** [Programa académico]
- **Asignatura o proyecto:** Proyecto de Grado / PGC
- **Tipo de proyecto:** Aplicación web local con Machine Learning
- **Tecnología principal:** Python, scikit-learn, FastAPI, HTML, CSS, JavaScript y NLP local basado en reglas

# 2. Problema y contexto

La identificación temprana de patrones de conducta infantil puede ser compleja, especialmente cuando la información se recoge con formularios técnicos que no siempre resultan claros para cuidadores o usuarios sin formación en psicología. Este proyecto trabaja con dos fuentes: un dataset estructurado con cinco dominios de trastornos y un archivo de preguntas humanizadas ya alineadas con los inputs del modelo.  

La solución permite que el usuario responda en lenguaje natural, traduce esas respuestas a variables estructuradas y aplica un modelo Random Forest para estimar compatibilidad de patrón. El sistema es completamente local, sin dependencias de APIs pagas o internet para inferencia, y no se presenta como herramienta diagnóstica.

# 3. Objetivos

## 3.1 Objetivo general

Desarrollar una aplicación web local basada en Random Forest que permita estimar, de manera preliminar y no diagnóstica, la compatibilidad de ciertos patrones de comportamiento infantil con trastorno de conducta, utilizando preguntas humanizadas y respuestas en lenguaje natural interpretadas mediante reglas locales.

## 3.2 Objetivos específicos

- Cargar e inspeccionar automáticamente el dataset principal.
- Cargar el diccionario de preguntas humanizadas.
- Validar la correspondencia entre inputs y preguntas.
- Seleccionar el target de conducta.
- Seleccionar features relevantes de conducta.
- Presentar preguntas entendibles para usuarios sin conocimientos de psicología.
- Interpretar respuestas en lenguaje natural con un parser local.
- Entrenar un modelo Random Forest.
- Optimizar el modelo priorizando F1-score y recall.
- Ajustar un threshold de decisión usando validación.
- Evaluar el modelo con métricas de clasificación.
- Crear una interfaz web local profesional con HTML, CSS y JavaScript.
- Registrar auditoría de decisiones, respuestas e inferencias.
- Generar evidencias para presentación y documento PGC.

# 4. Metodología

Fases implementadas:

1. Ubicación de ambos CSV en `data/`.
2. Análisis de contrato entre dataset y cuestionario.
3. Selección de target.
4. Selección de features.
5. Preprocesamiento.
6. Uso de preguntas humanizadas.
7. Implementación de parser NLP local.
8. Entrenamiento con Random Forest.
9. Optimización con prioridad F1 y recall.
10. Ajuste de threshold.
11. Evaluación con test independiente.
12. Desarrollo de backend FastAPI.
13. Desarrollo de frontend HTML/CSS/JS.
14. Validación y pruebas.
15. Generación de documentación.

# 5. Diseño de solución

Arquitectura principal:

- Dataset principal → Validación de contrato → Selección de target/features → Preprocesamiento → Random Forest → Optimización F1/Recall → Threshold → Métricas.
- CSV de preguntas → Preguntas humanizadas → Interfaz web → Respuestas naturales → Parser local → Variables → Predicción → Explicación clara.

Componentes:

- `data_contract.py`
- `data_loader.py`
- `questionnaire_loader.py`
- `feature_selection.py`
- `preprocessing.py`
- `question_generator.py`
- `question_explainer.py`
- `nlp_interpreter.py`
- `model.py`
- `predictor.py`
- `web_app.py`
- `index.html`
- `styles.css`
- `app.js`
- `audit.py`

# 6. Desarrollo

Tecnologías usadas:

- Python
- pandas
- numpy
- scikit-learn
- RandomForestClassifier
- FastAPI
- Uvicorn
- HTML
- CSS
- JavaScript
- rapidfuzz
- matplotlib
- joblib
- python-docx

Justificación de Random Forest:

- Es robusto para datos tabulares.
- Permite interpretabilidad por importancia de variables.
- No requiere GPU.
- Es liviano para ejecución local.
- Es adecuado para prototipos académicos auditables.

Justificación de F1 y recall:

- F1 equilibra precision y recall.
- Recall reduce el riesgo de omitir casos potencialmente relevantes.
- Precision se reporta para vigilar falsos positivos.

Proceso implementado:

- carga de datasets,
- validación de correspondencia,
- selección del target,
- selección de features,
- entrenamiento,
- optimización de threshold,
- evaluación,
- interfaz web,
- auditoría.

Aclaración técnica:

Random Forest no usa épocas como una red neuronal. En este proyecto se documentan número de árboles, profundidad, balanceo de clases, matriz de confusión, métricas, threshold e importancia de variables.

# 7. Validación

Se implementaron:

- pruebas unitarias,
- validación del contrato de datos,
- validación de parser,
- validación de carga de dataset,
- validación de predicción,
- validación de endpoints FastAPI,
- validación visual básica de frontend,
- revisión de métricas,
- matriz de confusión,
- curva de threshold,
- importancia de variables,
- pruebas manuales con respuestas claras y ambiguas.

# 8. Resultados

- Dataset usado: `hybrid_no_external_scores_dataset_ready(1).csv` o `hybrid_no_external_scores_dataset_ready.csv`
- Cuestionario usado: `questionnaire_v16_4_visible_questions_excel_utf8.csv`
- Target: `target_domain_conduct_final`
- Número de registros: [n]
- Número de features usados: [n]
- Threshold final: [valor]
- Accuracy: [valor]
- Precision: [valor]
- Recall: [valor]
- F1: [valor]
- ROC-AUC: [valor si aplica]
- PR-AUC: [valor si aplica]

# 9. Conclusiones

- Se desarrolló una herramienta local y gratuita.
- El sistema usa preguntas humanizadas ya relacionadas con los inputs.
- El usuario puede responder en lenguaje natural.
- Random Forest permitió una solución liviana e interpretable.
- La optimización priorizó F1 y recall.
- La interfaz web local permite una experiencia más profesional que interfaces Python autogeneradas.
- La auditoría mejora trazabilidad.
- La herramienta no sustituye evaluación clínica.

# 10. Fuentes utilizadas

- Documentación oficial de scikit-learn.
- Documentación oficial de FastAPI.
- Documentación oficial de pandas.
- Documentación oficial de Uvicorn.
- Literatura sobre Random Forest.
- Fuentes sobre uso responsable de IA en salud.
- Dataset utilizado: [citar fuente del dataset si aplica].

# 11. Anexos

- enlace a repositorio,
- capturas de interfaz,
- matriz de confusión,
- importancia de variables,
- curva de threshold,
- métricas,
- logs de auditoría,
- análisis de contrato de datos,
- instrucciones de ejecución.


## Actualización de implementación (versión chat)

- Target por defecto: 	arget_domain_conduct_final.
- Filtrado de preguntas por dominio objetivo para evitar fuga de ADHD cuando el target es conducta.
- Frontend en modo oscuro tipo chat con carga automática de preguntas.
- NLP local por reglas con inferencia de escala (inary, 	emporal_0_2, requency_0_3, observation_0_2, impact_0_3, 
umeric_range, categorical).
- Respuestas ambiguas generan aclaraciones específicas por escala.
- El sistema evita mostrar nombres técnicos y JSON crudo en la interfaz principal.

