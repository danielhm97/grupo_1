# Reporte Técnico — SVM Diabetes Prediction

## 1. Dataset

**Fuente**: `data/diabetes_prediction_dataset.csv` (100 000 registros)

**Target**: diabetes binaria — 91.5 % sin diabetes, 8.5 % con diabetes.

**Features**: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level

### Limpieza

- BMI >= 65 eliminados
- gender == 'Other' eliminado
- age redondeado a enteros
- Resultado: 99 935 muestras

### Smoking history

Mapeo: No Info→NoInfo, never→NeverSmoked, ever/not current/former→HasSmoked, current→Smoking

---

## 2. Preprocesamiento

| Tipo | Variables | Transformación |
|------|-----------|----------------|
| Numéricas | age, bmi | StandardScaler |
| Categóricas | gender, smoking_history | OneHotEncoder(drop='first') |
| Binarias | hypertension, heart_disease | passthrough |

Features resultantes: 8 (sin HbA1c/glucosa) / 10 (con HbA1c/glucosa).

### División train/test (80/20 estratificado)

Train: 79 948 (8.49 % diabetes), Test: 19 987 (8.49 % diabetes)

---

## 3. Data Leakage

HbA1c y blood_glucose eliminadas porque el diagnóstico de diabetes se basa directamente en HbA1c >= 6.5 % y glucosa >= 126 mg/dL (criterio ADA).

---

## 4. Modelo

SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42)

### GridSearch (scoring=average_precision, 3-fold, 24 combos)

| Parámetro | Valores |
|-----------|---------|
| C | [0.1, 0.5, 1.0] |
| gamma | ['scale', 0.01] |
| class_weight | ['balanced', {0:1, 1:5}, {0:1, 1:10}] |

---

## 5. Resultados CON HbA1c/glucosa

### Baseline (SVM_pipeline_con_hba1c_glucose.ipynb)

| Métrica | Train | Test |
|---------|-------|------|
| Accuracy | 0.8931 | 0.8949 |
| Precision | 0.4384 | 0.4433 |
| Recall | 0.9204 | 0.9287 |
| F1 | 0.5939 | 0.6002 |
| ROC-AUC | 0.9734 | 0.9736 |

Matriz test: [[16311, 1979], [121, 1576]]

### Optimizado (SVM_GridSearchCV_con_hba1c_glucose.ipynb)

Mejores params: C=1, class_weight={0:1, 1:5}, gamma='scale'
Mejor Average Precision (CV): 0.8307

| Métrica | Baseline | Optimizado | Mejora |
|---------|----------|------------|--------|
| Precision | 0.4433 | 0.6358 | +0.1925 |
| Recall | 0.9287 | 0.8167 | -0.1120 |
| F1 | 0.6002 | 0.7150 | +0.1148 |
| PR-AUC | 0.8526 | 0.8558 | +0.0032 |
| ROC-AUC | 0.9736 | 0.9701 | -0.0035 |

Matriz: [[17496, 794], [311, 1386]]

### Threshold tuning (Pipeline)

Threshold óptimo (max F2): 0.23

| Métrica | Baseline (0.5) | Óptimo (0.23) | Cambio |
|---------|---------------|---------------|--------|
| Precision | 0.4433 | 0.5654 | +0.1221 |
| Recall | 0.9287 | 0.8503 | -0.0784 |
| F1 | 0.6002 | 0.6792 | +0.0791 |

Matriz (th=0.23): [[17181, 1109], [254, 1443]]

### Threshold tuning (GridSearch)

Threshold óptimo: 0.12, Precision=0.6143, Recall=0.8268, F1=0.7048

---

## 6. Resultados SIN HbA1c/glucosa

### Baseline (baseline_sin_hba1c_glucose_executed.ipynb)

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.7285 |
| Precision | 0.2117 |
| Recall | 0.8067 |
| F1 | 0.3354 |
| PR-AUC | 0.2725 |
| ROC-AUC | 0.8247 |

Matriz: TN=13192, FP=5098, FN=328, TP=1369

### Optimizado (gridsearch_sin_hba1c_glucose_executed.ipynb)

Mejores params: C=0.5, gamma=0.01, cw={0: 1, 1: 5}
Mejor Average Precision (CV): 0.3127

| Métrica | Baseline (propio) | Optimizado | Mejora |
|---------|-------------------|------------|--------|
| Precision | 0.2036 | 0.2999 | 0.0963 |
| Recall | 0.8161 | 0.4797 | -0.3365 |
| F1 | 0.3259 | 0.3691 | 0.0432 |
| PR-AUC | 0.2520 | 0.3075 | 0.0555 |
| ROC-AUC | 0.8175 | 0.8314 | 0.0139 |

Matriz: TN=16390, FP=1900, FN=883, TP=814

### Threshold tuning

Threshold óptimo (max F1): 0.10

| Métrica | Default (0.5) | Óptimo (0.10) |
|---------|---------------|---------------|
| Precision | 0.2999 | 0.2710 |
| Recall | 0.4797 | 0.5875 |
| F1 | 0.3691 | 0.3709 |

---

## 7. SHAP

KernelExplainer. Para CON HbA1c se entrena SVM pequeño en 10000 muestras (2s, 1013 SV, SHAP en 33s).

Features (8 sin HbA1c): num__age, num__bmi, cat__gender_Male, cat__smoking_history_NeverSmoked, cat__smoking_history_NoInfo, cat__smoking_history_Smoking, bin__hypertension, bin__heart_disease
