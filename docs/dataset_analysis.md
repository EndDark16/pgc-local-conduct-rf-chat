# Análisis de Contrato de Datos

## Archivos analizados
- Dataset principal: `hybrid_no_external_scores_dataset_ready(1).csv`
- Cuestionario humanizado: `questionnaire_v16_4_visible_questions_excel_utf8.csv`

## Dimensiones
- Dataset principal: `[2400, 191]`
- Cuestionario: `[146, 62]`

## Relación input-pregunta
- Features del cuestionario: `146`
- Features coincidentes con dataset: `146`
- Features del cuestionario no presentes en dataset: `0`

## Targets detectados
- Targets: `['target_domain_adhd_final', 'target_domain_conduct_final', 'target_domain_elimination_final', 'target_domain_anxiety_final', 'target_domain_depression_final']`
- Target seleccionado: `target_domain_conduct_final`
- Método de selección: `default_conduct`

## Distribución del target seleccionado
`{0: 1602, 1: 798}`

## Features candidatas de conducta
Total: `23`

## Advertencias
- Sin advertencias críticas.

## Decisiones tomadas
- El cuestionario se usa como contrato oficial de entradas.
- Se prioriza target_domain_conduct_final si está disponible.
- Las preguntas humanizadas se toman del CSV del cuestionario.
