# SVM Diabetes Prediction

Clasificación SVM para predicción de diabetes con experimentos **con y sin** HbA1c/glucosa.

## Quick Start

```bash
papermill SVM_model/baseline_sin_hba1c_glucose.ipynb sin_executed.ipynb
papermill SVM_model/gridsearch_sin_hba1c_glucose.ipynb gs_executed.ipynb
```

## Notebooks

| Notebook | Descripción |
|----------|-------------|
| `baseline_sin_hba1c_glucose.ipynb` | Baseline SVM sin HbA1c/glucosa + SHAP |
| `gridsearch_sin_hba1c_glucose.ipynb` | GridSearch CV + SHAP + threshold tuning |
| `SVM_pipeline_con_hba1c_glucose.ipynb` | Tutorial paso a paso CON HbA1c/glucosa |
| `SVM_GridSearchCV_con_hba1c_glucose.ipynb` | GridSearch CV CON HbA1c/glucosa |

## Resultados Clave

| Experimento | PR-AUC | F1 | Recall |
|-------------|--------|----|--------|
| CON HbA1c/glucosa — Baseline | 0.8526 | 0.6002 | 0.9287 |
| CON HbA1c/glucosa — Optimizado | **0.856** | **0.715** | 0.817 |
| SIN HbA1c/glucosa — Baseline | 0.2520 | 0.3259 | **0.8161** |
| SIN HbA1c/glucosa — Optimizado | 0.3075 | 0.3691 | 0.4797 |

→ Ver `report.md` para justificación completa de decisiones y resultados.

## Archivos

```
SVM_model/
├── pipeline.py              # 7 funciones modulares
├── baseline_sin_*.ipynb     # Experimentos sin HbA1c/glucosa
├── gridsearch_sin_*.ipynb
├── SVM_pipeline_con_*.ipynb # Experimentos con HbA1c/glucosa
├── SVM_GridSearchCV_con_*.ipynb
├── artifacts/                # Modelos .pkl y métricas .json
├── README.md                 # Este archivo
└── report.md                 # Justificación completa
```

## Autores

Daniel Huarita — David Moya — Iñaki Asúa — Juan María Jiménez
