# 1. Portada

- **Título:** Sistema local de apoyo preliminar para estimar patrones compatibles con trastorno de conducta mediante Random Forest
- **Integrantes:** [nombres]
- **Programa:** [programa]
- **Contexto:** Proyecto PGC - Aplicación web local de Machine Learning

# 2. Problema

- Formularios técnicos pueden ser difíciles para usuarios no expertos.
- Los términos psicológicos pueden generar confusión.
- Se cuenta con un archivo de preguntas humanizadas asociado a cada input.
- Se requiere una herramienta que convierta respuestas naturales en variables del modelo.
- Se busca priorizar F1 y recall para reducir omisiones relevantes.
- El sistema debe ser local, gratuito, visualmente profesional y auditable.
- No reemplaza diagnóstico médico.

# 3. Objetivo

Desarrollar una aplicación web local que use preguntas humanizadas, interprete respuestas en lenguaje natural, las convierta en variables estructuradas y use Random Forest optimizado por F1 y recall para estimar patrones compatibles con trastorno de conducta.

# 4. Metodología

1. Dataset principal en `data/`.
2. Cuestionario humanizado en `data/`.
3. Validación de correspondencia input-pregunta.
4. Selección de target de conducta.
5. Selección de features.
6. Interpretación local de lenguaje natural.
7. Entrenamiento de Random Forest.
8. Optimización F1/recall.
9. Ajuste de threshold.
10. Evaluación.
11. Backend FastAPI.
12. Frontend HTML/CSS/JS.
13. Auditoría.

# 5. Diseño

- Dataset principal → Preprocesamiento → Random Forest → Threshold optimizado → Métricas.
- CSV de preguntas → Preguntas simples → Parser NLP local → Variables → Predicción → Resultado claro.
- Frontend HTML/CSS/JS → FastAPI local → Modelo entrenado → Respuesta al usuario.

# 6. Demo

Guion:

1. Abrir terminal.
2. Ejecutar `python validate_project.py`.
3. Mostrar reporte de contrato entre datasets.
4. Ejecutar `python train.py`.
5. Mostrar métricas, F1, recall y threshold.
6. Ejecutar `python app.py`.
7. Abrir `http://127.0.0.1:8000`.
8. Mostrar diseño de la interfaz.
9. Responder una pregunta clara.
10. Responder una pregunta ambigua.
11. Mostrar cómo el sistema pide aclaración.
12. Usar botón “No entiendo la pregunta”.
13. Confirmar respuestas.
14. Generar estimación final.
15. Mostrar advertencia de no diagnóstico.
16. Mostrar logs, matriz de confusión o importancia de variables.

Plan de contingencia:

- Tener capturas.
- Tener video corto de 2 minutos.
- Tener modelo ya entrenado.
- Tener ambos CSV en `data/`.
- Tener entorno virtual instalado.
- Tener comandos en `docs/demo_commands.txt`.

# 7. Resultados

- Modelo entrenado: Random Forest.
- Dataset: `hybrid_no_external_scores_dataset_ready(1).csv` o `hybrid_no_external_scores_dataset_ready.csv`.
- Preguntas: `questionnaire_v16_4_visible_questions_excel_utf8.csv`.
- Target: `target_domain_conduct_final`.
- Threshold final: [valor].
- Accuracy: [valor].
- Precision: [valor].
- Recall: [valor].
- F1: [valor].
- Matriz de confusión generada.
- Curva de threshold generada.
- Importancia de variables generada.
- Interfaz web local funcional.
- Parser local funcionando sin internet.
- Auditoría local generada.

# 8. Conclusiones

- Se logró una solución local, gratuita y funcional.
- El usuario responde preguntas humanizadas.
- El sistema usa lenguaje natural sin APIs pagas.
- Random Forest permite una solución liviana e interpretable.
- La optimización prioriza F1 y recall.
- La interfaz HTML/CSS/JS mejora la presentación visual y la experiencia de usuario.
- La herramienta es preliminar y no diagnóstica.
- El sistema es auditable y replicable.


## Ajustes finales de producto

- Se fijó 	arget_domain_conduct_final como valor por defecto.
- Se filtraron preguntas por dominio para evitar incoherencias (ejemplo: preguntas ADHD en modo conducta).
- La experiencia final es conversacional en modo oscuro y con carga automática.
- El parser local interpreta lenguaje natural según el tipo de escala y solicita aclaración cuando corresponde.


## Nota de cierre técnico

- Target por defecto: `target_domain_conduct_final`.
- Preguntas filtradas por dominio del target (sin mezcla ADHD cuando el target es conducta).
- Parser NLP local por escalas con aclaración contextual.
- Flujo conversacional completo: preguntas, confirmación, impresión orientativa y preguntas post-resultado.
- Arquitectura 100% local, sin APIs externas ni servicios pagos.
