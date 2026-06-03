# Reporte Técnico — SVM Diabetes Prediction

## 1. Dataset

**Fuente**: `data/diabetes_prediction_dataset.csv` (100 000 registros)

**Target**: `diabetes` binaria — 91.5 % sin diabetes, 8.5 % con diabetes (desbalanceado).

**Features originales** (8):

| Feature | Tipo | Descripción |
|---------|------|-------------|
| gender | Categórica | Male / Female / Other |
| age | Numérica | Edad en años |
| hypertension | Binaria | 0/1 |
| heart_disease | Binaria | 0/1 |
| smoking_history | Categórica | 6 niveles |
| bmi | Numérica | Índice de masa corporal |
| HbA1c_level | Numérica | Hemoglobina glicosilada |
| blood_glucose_level | Numérica | Nivel de glucosa en sangre |

### 1.1. Limpieza

- **BMI >= 65**: 47 observaciones eliminadas (outliers extremos fisiológicamente inviables).
- **gender == 'Other'**: ~18 observaciones eliminadas (categoría residual sin significado clínico).
- **Redondeo de age** a enteros.
- **Resultado**: 99 935 muestras.

### 1.2. Mapeo de smoking_history

| Original | Nueva | Recuento | Diabetes % |
|----------|-------|----------|-----------|
| never | NeverSmoked | 35 053 | 7.7 % |
| No Info | NoInfo | 35 817 | 8.4 % |
| current | Smoking | 9 643 | 13.7 % |
| former / not current / ever | HasSmoked | 19 422 | 9.0 % |

Justificación: reducir dimensionalidad manteniendo diferencia entre fumadores activos (Smoking, 13.7 %) y el resto.

---

## 2. Preprocesamiento

| Tipo | Variables | Transformación | Justificación |
|------|-----------|----------------|---------------|
| Numéricas | age, bmi | StandardScaler | SVM requiere media 0, std 1 para distancias |
| Categóricas | gender, smoking_history | OneHotEncoder(drop='first') | Evita orden artificial (Label encoding), aprovecha kernel RBF |
| Binarias | hypertension, heart_disease | passthrough | Ya son 0/1, no requieren escalado |

**Features resultantes**:
- **CON HbA1c/glucosa**: 12 features (age, bmi, HbA1c_level, blood_glucose_level, gender_Male, smoking_history_NeverSmoked, smoking_history_NoInfo, smoking_history_Smoking, hypertension, heart_disease + 2 más de OneHot)
- **SIN HbA1c/glucosa**: 8 features (excluyendo HbA1c_level y blood_glucose_level)

### 2.1. División train/test

- test_size = 0.20, estratificado por target
- Train: 79 948 muestras, prevalencia 8.5 %
- Test: 19 987 muestras, prevalencia 8.5 %

---

## 3. Eliminación de HbA1c y blood_glucose — Data Leakage

**Decisión**: Eliminar `HbA1c_level` y `blood_glucose_level` por fuga de información (data leakage).

**Justificación clínica**: El diagnóstico de diabetes se basa directamente en estos valores:
- HbA1c >= 6.5 % → diabetes (criterio ADA)
- Glucosa en ayunas >= 126 mg/dL → diabetes

Incluirlos como predictores es equivalente a preguntar "¿tiene diabetes?" usando la respuesta. Esto infla artificialmente las métricas y no refleja la capacidad predictiva real del modelo con variables clínicas disponibles sin análisis de sangre.

**Impacto cuantitativo**:

| Métrica | CON HbA1c | SIN HbA1c | Diferencia |
|---------|-----------|-----------|------------|
| PR-AUC | 0.856 | 0.307 | -64 % |
| ROC-AUC | 0.975 | 0.831 | -15 % |
| F1 | 0.715 | 0.369 | -48 % |

---

## 4. Codificación de Variables Categóricas

**Decisión**: OneHotEncoder con drop='first'.

**Justificación**:
- SVM calcula distancias euclidianas. OneHot crea vectores ortogonales donde cada categoría está a distancia √2 de las demás, sin orden artificial.
- Label encoding introduce orden falso (NeverSmoked=0 < Smoking=1 < HasSmoked=2) que SVM interpreta como escala continua, degradando rendimiento.
- WOE reduce a 1 dimensión, no aprovecha la capacidad no-lineal del kernel RBF.
- Target encoding es viable pero introduce riesgo de overfitting al usar información del target.

---

## 5. Modelo SVM — Configuración

### 5.1. Baseline

```python
SVC(kernel='rbf', C=1.0, gamma='scale',
    class_weight='balanced', probability=True, random_state=42)
```

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| kernel='rbf' | No-lineal | Relaciones entre features y diabetes no son lineales |
| C=1.0 | Default | Punto de partida estándar |
| gamma='scale' | 1/(n_features * X.var()) | Adaptativo a la escala de los datos |
| class_weight='balanced' | Peso ~5.9 para clase 1 | Compensa desbalance 91.5/8.5 |
| probability=True | Habilita predict_proba | Necesario para PR-AUC, ROC-AUC, SHAP |

### 5.2. GridSearch CV

**Grid** (18 combinaciones):

| Parámetro | Valores |
|-----------|---------|
| C | [0.1, 0.5, 1.0] |
| gamma | ['scale', 0.01] |
| class_weight | ['balanced', {0:1, 1:5}, {0:1, 1:10}] |

**Configuración**:
- scoring = 'average_precision' (PR-AUC como métrica objetivo)
- cv = StratifiedKFold(3, shuffle=True) (3 folds, preserva proporción de clases)
- refit completo en 80k samples
- Tiempo estimado: ~12-15 min

**Métrica de optimización**: Average Precision (PR-AUC)

Justificación:
- PR-AUC evalúa todo el rango de umbrales, no solo un punto.
- Es la métrica estándar para clasificación desbalanceada con clase minoritaria relevante.
- Recall puro llevaría a clasificar todo como positivo (Precision=0).
- F1 evalúa un solo umbral, PR-AUC es más robusto.

---

## 6. Métricas de Evaluación

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| Recall | TP/(TP+FN) | De los diabéticos reales, cuántos detectamos. **Prioridad clínica**: un FN = paciente sin tratamiento |
| Precision | TP/(TP+FP) | De los que etiquetamos como diabetes, cuántos realmente lo son. FP = falsa alarma |
| F1 | 2·P·R/(P+R) | Media armónica. Balance entre Precision y Recall |
| PR-AUC | Área bajo curva P-R | **Métrica principal**. Evalúa rendimiento en todos los umbrales para clase minoritaria |
| ROC-AUC | Área bajo curva ROC | Capacidad discriminativa global. Menos informativo en desbalance |

---

## 7. Resultados

### 7.1. CON HbA1c_level y blood_glucose_level

**Baseline** (subset 20 %, ~16k muestras, ~38s entrenamiento):

| Métrica | Valor |
|---------|-------|
| Precision | 0.4433 |
| Recall | 0.9287 |
| F1 | 0.6002 |
| PR-AUC | 0.8526 |
| ROC-AUC | 0.9736 |

**Optimizado** (GridSearch con refit en 80k, ~12 min):

Mejores parámetros: C=1, class_weight={0:1, 1:5}, gamma='scale'

| Métrica | Baseline | Optimizado | Cambio |
|---------|----------|------------|--------|
| Precision | 0.4433 | 0.636 | +0.1927 |
| Recall | 0.9287 | 0.817 | -0.1117 |
| F1 | 0.6002 | 0.715 | +0.1148 |
| PR-AUC | 0.8526 | 0.856 | +0.0034 |
| ROC-AUC | 0.9736 | 0.975 | +0.0014 |

Interpretación: La optimización mejora significativamente Precision (+0.19) y F1 (+0.11) a costa de Recall (-0.11). El modelo baseline ya tenía recall excelente (92.9 %) con class_weight='balanced'.

### 7.2. SIN HbA1c_level y blood_glucose_level

**Baseline** (C=1, gamma='scale', class_weight='balanced', probability=True, subset 20 %):

| Métrica | Valor |
|---------|-------|
| Precision | 0.2036 |
| Recall | 0.8161 |
| F1 | 0.3259 |
| PR-AUC | 0.2520 |
| ROC-AUC | 0.8175 |

**Optimizado** (GridSearch con refit en 80k, ~20 min):

Mejores parámetros: C=0.5, gamma=0.01, class_weight={0:1, 1:5}

| Métrica | Baseline | Optimizado | Cambio |
|---------|----------|------------|--------|
| Precision | 0.2036 | 0.2999 | +0.0963 |
| Recall | 0.8161 | 0.4797 | -0.3364 |
| F1 | 0.3259 | 0.3691 | +0.0432 |
| PR-AUC | 0.2520 | 0.3075 | +0.0555 |
| ROC-AUC | 0.8175 | 0.8314 | +0.0139 |

Interpretación:
- Sin HbA1c/glucosa, el modelo pierde predictores muy informativos. PR-AUC cae de 0.856 a 0.307 (-64 %).
- El baseline mantiene Recall alto (81.6 %) incluso sin estas variables.
- La optimización mejora Precision (+0.096) y PR-AUC (+0.056) pero sacrifica Recall (-0.336).
- gamma=0.01 (menor que 'scale' ~0.125) indica que el modelo necesita kernels más suaves, probablemente porque las features restantes tienen menor señal.
- C=0.5 (menor que baseline 1.0) indica que más regularización ayuda a evitar overfitting cuando la señal es más débil.

**Threshold tuning** (optimizado, umbral óptimo encontrado = 0.10):

| Métrica | Default (0.5) | Óptimo (0.10) |
|---------|---------------|----------------|
| Precision | 0.2999 | 0.2710 |
| Recall | 0.4797 | 0.5875 |
| F1 | 0.3691 | 0.3709 |

Bajar el umbral a 0.10 recupera +0.1078 de Recall con pérdida mínima de Precision (-0.0289). El F1 se mantiene prácticamente igual.

### 7.3. Comparativa Con vs Sin HbA1c/glucosa

| Métrica | CON (opt) | SIN (opt) | Diferencia | % Cambio |
|---------|-----------|-----------|------------|----------|
| Precision | 0.636 | 0.2999 | -0.3361 | -53 % |
| Recall | 0.817 | 0.4797 | -0.3373 | -41 % |
| F1 | 0.715 | 0.3691 | -0.3459 | -48 % |
| PR-AUC | 0.856 | 0.3075 | -0.5485 | -64 % |
| ROC-AUC | 0.975 | 0.8314 | -0.1436 | -15 % |

---

## 8. SHAP — Explicabilidad

Se aplicó SHAP con **KernelExplainer** en todos los notebooks.

**Problema**: KernelExplainer sobre el SVM grande (entrenado en 80k muestras) es extremadamente lento (>15 min para 3 muestras de explicación) debido al alto número de support vectors.

**Solución**: Para los notebooks CON HbA1c/glucosa, se entrena un SVM pequeño dedicado (SVC con RBF en 2000 muestras aleatorias) solo para el análisis SHAP. Esto reduce el tiempo de ~15 min a ~30-60s.

**Features analizadas** (8, experimentos sin HbA1c/glucosa):
`num__age`, `num__bmi`, `cat__gender_Male`, `cat__smoking_history_NeverSmoked`, `cat__smoking_history_NoInfo`, `cat__smoking_history_Smoking`, `bin__hypertension`, `bin__heart_disease`

**Features analizadas** (12, experimentos con HbA1c/glucosa):
Las 8 anteriores + `num__HbA1c_level`, `num__blood_glucose_level`, más features adicionales de OneHot.

**Plots generados** (por notebook):
1. Summary plot (barra): importancia global de features
2. Beeswarm plot: distribución del impacto de cada feature

---

## 9. Análisis de Parámetros Óptimos

### Con HbA1c/glucosa: C=1, gamma='scale', cw={0:1, 1:5}

- **C=1**: el modelo con HbA1c/glucosa tiene señal fuerte; no necesita regularización adicional.
- **gamma='scale'**: las features con HbA1c/glucosa tienen varianza informativa; 'scale' (~0.083 para 12 features) permite influencia local adecuada.
- **cw={0:1, 1:5}}: peso 5 para diabetes, balanceado suficiente (el baseline con 'balanced' daba peso ~5.9).

### Sin HbA1c/glucosa: C=0.5, gamma=0.01, cw={0:1, 1:5}

- **C=0.5**: menor que baseline (1.0). Más regularización necesaria porque la señal es más débil y el modelo tiende a sobreajustar.
- **gamma=0.01**: mucho menor que 'scale' (~0.125). Kernels muy suaves porque las features restantes (age, bmi, género, smoking, hypertension, heart_disease) tienen relaciones más sutiles con el target.
- **cw={0:1, 1:5}**: mismo peso que con HbA1c. El desbalance se mantiene constante.

---

## 10. Conclusiones

1. **HbA1c y blood_glucose son predictores artificialmente perfectos**: su inclusión infla métricas (PR-AUC 0.856 vs 0.307) y no representa la capacidad predictiva real del modelo con datos clínicos básicos.

2. **Sin HbA1c/glucosa, el modelo tiene rendimiento limitado pero útil**: PR-AUC 0.307, Recall 81.6 % en baseline. El SVM puede capturar algo de señal de edad, BMI, hipertensión y hábito de fumar.

3. **class_weight={0:1, 1:5} fue la configuración más robusta**: aparece como óptimo en ambos experimentos y escala correctamente para el desbalance 91.5/8.5.

4. **Gamma menor (0.01) y C menor (0.5) para modelo sin HbA1c/glucosa**: señal más débil requiere kernels más suaves y más regularización.

5. **OneHotEncoder + StandardScaler es la configuración correcta para SVM**: evita orden artificial y maximiza el aprovechamiento del kernel RBF.

6. **PR-AUC como métrica de optimización**: más informativa que F1 o Recall puro para clasificación desbalanceada con clase minoritaria relevante.

7. **SHAP con SVM pequeño dedicado**: solución práctica al problema de lentitud de KernelExplainer en SVMs grandes, manteniendo la explicabilidad del modelo.
