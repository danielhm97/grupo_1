import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (recall_score, roc_auc_score, average_precision_score,
                             f1_score, confusion_matrix)
import code.utils_pipeline_dhm as up


# ---- Data ----
data = pd.read_csv('./data/diabetes_prediction_dataset.csv', sep=',')
data = data[data['bmi'] < 65]
data = data[data['gender'] != 'Other']
smoking_mapping = {
    'No Info': 'No Info', 'never': 'never', 'former': 'former',
    'current': 'current', 'not current': 'former', 'ever': 'former'
}

X = data.drop(columns=["diabetes"])
y = data["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

binary_cols = ['hypertension', 'heart_disease']
column_config = {
    'gender':               {'transform': 'woe'},
    'smoking_history':      {'transform': 'woe'},
    'hypertension':         {'transform': 'woe'},
    'heart_disease':        {'transform': 'woe'},
    'age':                  {'transform': 'binning', 'max_bins': 8},
    'bmi':                  {'transform': 'binning', 'max_bins': 8},
    'HbA1c_level':         {'transform': 'binning', 'max_bins': 5},
    'blood_glucose_level':  {'transform': 'binning', 'max_bins': 4},
}
all_features = list(column_config.keys())

# ---- Pipeline (column-agnostic preprocessor) ----
pipe = Pipeline([
    ("type_cast", up.TypeCaster(columns=binary_cols, dtype=str)),
    ("smoke_map", up.ColumnMapper(column='smoking_history', mapping=smoking_mapping)),
    ("preprocessor", up.ColumnConfigPreprocessor(column_config)),
    ("model", LogisticRegression(class_weight='balanced', random_state=42))
])

# ---- GridSearchCV for C ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
param_grid = {"model__C": [0.001, 0.01, 0.1, 1.0]}

grid = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring, refit="pr_auc", n_jobs=-1)
grid.fit(X_train, y_train)

# Results
cv_results = pd.DataFrame(grid.cv_results_)
cv_results = cv_results[["param_model__C", "mean_test_roc_auc", "mean_test_pr_auc",
                         "std_test_roc_auc", "std_test_pr_auc"]]
print(cv_results, '\n')

best_params = grid.best_params_
best_score = grid.best_score_
print("Stats for best parameters:")
print("Best Parameters:", best_params)
print("Mean PR-AUC Score:", best_score, '\n')

# ---- 8-feature evaluation ----
best_model = grid.best_estimator_
train_preds = best_model.predict_proba(X_train)[:, 1]
thresholds = np.linspace(0.01, 0.99, 99)
opt_t = thresholds[np.argmax([f1_score(y_train, (train_preds >= t).astype(int)) for t in thresholds])]
print(f'Optimal threshold (max F1 en train): {opt_t:.3f}\n')

print("Train Metrics for the Best Model:")
print('Train PR-AUC:', average_precision_score(y_train, train_preds))
print('Train ROC-AUC:', roc_auc_score(y_train, train_preds))
print('Train Recall:', recall_score(y_train, (train_preds >= opt_t).astype(int)), '\n')

print("Test Metrics for the Best Model:")
test_preds = best_model.predict_proba(X_test)[:, 1]
print('Test PR-AUC:', average_precision_score(y_test, test_preds))
print('Test ROC-AUC:', roc_auc_score(y_test, test_preds))
print('Test Recall:', recall_score(y_test, (test_preds >= opt_t).astype(int)))

# Extract coefficients
model = best_model.named_steps['model']
fnames = best_model.named_steps['preprocessor'].get_feature_names_out()
coef_df = pd.DataFrame({
    'feature': fnames,
    'coef': model.coef_[0],
    'odds_ratio': np.exp(model.coef_[0])
}).sort_values('coef', ascending=False)
print('\nCoeficientes del modelo (8 features):')
print(coef_df.to_string(index=False))

# ---- Backward Elimination ----
best_C = grid.best_params_['model__C']
print(f'\n=== Backward Elimination (best C={best_C}) ===')

remaining = all_features.copy()
backward_results = []
cv_sel = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate full set (8 features)
pr_m, pr_s, roc_m, roc_s = up.evaluate(
    up.build_pipe(remaining, best_C, column_config, binary_cols, smoking_mapping),
    X_train, y_train, cv_sel)
backward_results.append({
    'n_features': 8, 'drop': '(none)', 'features': ', '.join(sorted(remaining)),
    'pr_auc': f'{pr_m:.4f} ±{pr_s:.4f}', 'roc_auc': f'{roc_m:.4f} ±{roc_s:.4f}'
})
print(f'  8 features | PR-AUC: {pr_m:.4f} ±{pr_s:.4f} | ROC-AUC: {roc_m:.4f} ±{roc_s:.4f}')

while len(remaining) > 1:
    best_auc = -1.0
    best_candidate = None

    for candidate in remaining:
        temp = [f for f in remaining if f != candidate]
        pr_m_temp, _, _, _ = up.evaluate(
            up.build_pipe(temp, best_C, column_config, binary_cols, smoking_mapping),
            X_train, y_train, cv_sel)
        if pr_m_temp > best_auc:
            best_auc = pr_m_temp
            best_candidate = candidate

    remaining.remove(best_candidate)
    pr_m, pr_s, roc_m, roc_s = up.evaluate(
        up.build_pipe(remaining, best_C, column_config, binary_cols, smoking_mapping),
        X_train, y_train, cv_sel)
    backward_results.append({
        'n_features': len(remaining),
        'drop': best_candidate,
        'features': ', '.join(sorted(remaining)),
        'pr_auc': f'{pr_m:.4f} ±{pr_s:.4f}',
        'roc_auc': f'{roc_m:.4f} ±{roc_s:.4f}'
    })
    print(f'  Drop {best_candidate:20s} -> {len(remaining)} feat | PR-AUC: {pr_m:.4f} +/-{pr_s:.4f} | ROC-AUC: {roc_m:.4f} +/-{roc_s:.4f}')

# Print summary table
print('\nBackward elimination summary:')
print(pd.DataFrame(backward_results).to_string(index=False))

# Select optimal subset via elbow ratio
be_df = pd.DataFrame(backward_results)
be_df['pr_val'] = be_df['pr_auc'].str.extract(r'([\d.]+)').astype(float)
losses = be_df['pr_val'].diff().abs().dropna()
ratios = losses / losses.shift(1)
elbow_idx = ratios.dropna().idxmax()
optimal_n = int(be_df.loc[elbow_idx, 'n_features'] + 1)
selected = be_df[be_df['n_features'] == optimal_n]['features'].iloc[0].split(', ')
print(f'\nOptimal subset via elbow ratio: {len(selected)} features')
print(f'Selected features: {selected}')

# ---- Final evaluation on selected subset ----
X_train_sel = X_train[selected]
X_test_sel = X_test[selected]

final_pipe = up.build_pipe(selected, best_C, column_config, binary_cols, smoking_mapping)

final_pipe.fit(X_train_sel, y_train)

print(f'\n=== Final evaluation ({len(selected)} features) ===')
for name, X, y in [('Train', X_train_sel, y_train), ('Test', X_test_sel, y_test)]:
    preds = final_pipe.predict_proba(X)[:, 1]
    opt_t = thresholds[np.argmax([f1_score(y, (preds >= t).astype(int)) for t in thresholds])]
    print(f'{name:6s} | PR-AUC: {average_precision_score(y, preds):.4f} | '
          f'ROC-AUC: {roc_auc_score(y, preds):.4f} | '
          f'Recall: {recall_score(y, (preds >= opt_t).astype(int)):.4f}')

preds_test = final_pipe.predict_proba(X_test_sel)[:, 1]
cm_t = thresholds[np.argmax([f1_score(y_test, (preds_test >= t).astype(int)) for t in thresholds])]
cm = confusion_matrix(y_test, (preds_test >= cm_t).astype(int))
print(f'\nMatriz de confusion (Test, threshold={cm_t:.3f}):')
print(f'{"":12s} Pred:0   Pred:1')
print(f'{"Actual 0":12s} {cm[0,0]:5d}    {cm[0,1]:5d}')
print(f'{"Actual 1":12s} {cm[1,0]:5d}    {cm[1,1]:5d}')

# ---- Interpretabilidad: WOE mappings y train transformado ----
print('\n' + '='*70)
print('=== WOE Mappings (interpretabilidad) ===')
print('='*70)
preproc = final_pipe.named_steps['preprocessor']
mappings_df = preproc.get_mappings()
print(mappings_df.to_string(index=False))

print('\n' + '='*70)
print('=== Train transformado (primeras 10 filas) ===')
print('='*70)
# Pasa por los pasos previos (type_cast, smoke_map) antes del preprocessor
X_train_tf = final_pipe[:-1].transform(X_train_sel)
print(X_train_tf.head(10).to_string())
