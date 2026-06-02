import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import category_encoders as ce
from optbinning import OptimalBinning


# ── Custom transformers ──

class TypeCaster(BaseEstimator, TransformerMixin):
    def __init__(self, columns, dtype):
        self.columns = columns
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        active = [c for c in self.columns if c in X.columns]
        if active:
            X[active] = X[active].astype(self.dtype)
        return X


class ColumnMapper(BaseEstimator, TransformerMixin):
    def __init__(self, column, mapping):
        self.column = column
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.column in X.columns:
            X[self.column] = X[self.column].map(self.mapping)
        return X


class ColumnConfigPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor con configuracion explicita por columna.

    column_config: dict
        key = nombre de columna
        value = dict con:
            'transform': 'woe' | 'binning'
            'max_bins': int (solo para binning, default 5)
    """
    def __init__(self, column_config):
        self.column_config = column_config

    def fit(self, X, y):
        self.col_order_ = list(self.column_config.keys())
        self.woe_encoders_ = {}
        self.binners_ = {}

        for col, cfg in self.column_config.items():
            if cfg['transform'] == 'woe':
                encoder = Pipeline([
                    ("woe", ce.WOEEncoder(cols=[col])),
                    ("invert", FunctionTransformer(lambda X: -X, feature_names_out="one-to-one"))
                ])
                encoder.fit(X[[col]], y)
                self.woe_encoders_[col] = encoder
            elif cfg['transform'] == 'binning':
                max_bins = cfg.get('max_bins', 5)
                optb = OptimalBinning(name=col, dtype="numerical", max_n_bins=max_bins)
                optb.fit(X[col].values, y)
                self.binners_[col] = optb

        return self

    def transform(self, X):
        result = pd.DataFrame(index=X.index)
        for col, cfg in self.column_config.items():
            if cfg['transform'] == 'woe':
                result[col] = self.woe_encoders_[col].transform(X[[col]])[col]
            elif cfg['transform'] == 'binning':
                result[col] = self.binners_[col].transform(X[col].values)
        return result

    def get_feature_names_out(self, input_features=None):
        return list(self.col_order_)

    def transform_to_df(self, X):
        return self.transform(X)

    def get_mappings(self):
        """DataFrame unico con mappings de WOE y binning por columna.

        Columnas: feature, category_bin, transformation, woe_raw, woe_model, IV
        """
        rows = []
        for col, cfg in self.column_config.items():
            if cfg['transform'] == 'woe':
                inner = self.woe_encoders_[col].named_steps['woe']
                for cm in inner.ordinal_encoder.category_mapping:
                    if cm['col'] != col:
                        continue
                    for cat, ord_val in cm['mapping'].items():
                        raw = inner.mapping[col][ord_val]
                        rows.append({
                            'feature': col,
                            'category_bin': cat,
                            'transformation': 'woe',
                            'woe_raw': raw, 'woe_model': -raw, 'IV': None
                        })
            elif cfg['transform'] == 'binning':
                tbl = self.binners_[col].binning_table.build()
                tbl = tbl[tbl['Bin'].astype(str).str.strip() != '']
                for _, r in tbl.iterrows():
                    rows.append({
                        'feature': col,
                        'category_bin': r['Bin'],
                        'transformation': 'binning',
                        'woe_raw': r['WoE'],
                        'woe_model': r['WoE'],
                        'IV': r['IV']
                    })
        return pd.DataFrame(rows)


# ── Pipeline builder ──

def build_pipe(features, best_C, column_config, binary_cols, smoking_mapping):
    """Construye pipeline completo para un subconjunto de features."""
    cfg = {k: v for k, v in column_config.items() if k in features}
    return Pipeline([
        ("type_cast", TypeCaster(columns=binary_cols, dtype=str)),
        ("smoke_map", ColumnMapper(column='smoking_history', mapping=smoking_mapping)),
        ("preprocessor", ColumnConfigPreprocessor(cfg)),
        ("model", LogisticRegression(C=best_C, class_weight='balanced', random_state=42))
    ])


# ── Evaluator ──

def evaluate(pipe, X_train, y_train, cv):
    """Evalua un pipeline con cross-validation y devuelve medias + stds."""
    scores = cross_validate(pipe, X_train, y_train, cv=cv,
                            scoring={'pr_auc': 'average_precision', 'roc_auc': 'roc_auc'},
                            n_jobs=-1)
    return (scores['test_pr_auc'].mean(), scores['test_pr_auc'].std(),
            scores['test_roc_auc'].mean(), scores['test_roc_auc'].std())
