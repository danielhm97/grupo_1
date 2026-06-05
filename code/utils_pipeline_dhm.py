import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import category_encoders as ce
from optbinning import OptimalBinning
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import plotnine as p9
from plotnine import ggplot


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


# ── Plotting functions ──

def plot_confusion_matrix(cm, title=None, fig_size=(4, 3)):
    """Heatmap de matriz de confusion con counts y % fila."""
    n = cm.shape[0]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0)

    fig, ax = plt.subplots(figsize=fig_size)
    cmap = np.full((n, n, 4), [0.8, 0.2, 0.2, 0.25], dtype=float)
    for i in range(n):
        cmap[i, i] = [0.2, 0.6, 0.2, 0.25]
    ax.imshow(cmap)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)',
                    ha='center', va='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(['Pred: 0', 'Pred: 1'])
    ax.set_yticklabels(['Actual 0', 'Actual 1'])
    ax.set_xlabel('Prediccion', fontsize=9)
    ax.set_ylabel('Real', fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_elbow(be_df, optimal_n, fig_size=(6, 4)):
    """Linea de PR-AUC vs n_features con marcador en el codo."""
    df = be_df.copy()
    line_col = 'steelblue'

    if 'pr_val' not in df.columns:
        df['pr_val'] = df['pr_auc'].str.extract(r'([\d.]+)').astype(float)
    p = (
        p9.ggplot(df, p9.aes(x='n_features', y='pr_val'))
        + p9.geom_line(color=line_col, size=0.8)
        + p9.geom_point(color=line_col, size=2)
        + p9.geom_point(
            data=df[df['n_features'] == optimal_n],
            mapping=p9.aes(x='n_features', y='pr_val'),
            color='red', size=3.5
        )
        + p9.geom_vline(xintercept=optimal_n, linetype='dashed',
                         color='red', size=0.5)
        + p9.labs(title='Backward Elimination',
                  x='Numero de features', y='PR-AUC (CV mean)')
        + p9.scale_x_continuous(
            breaks=sorted(df['n_features'].unique()),
            limits=(df['n_features'].min() - 0.3,
                    df['n_features'].max() + 0.3)
        )
        + p9.theme(
            panel_background=p9.element_rect(fill="#ffffff"),
            plot_background=p9.element_rect(fill='#ffffff'),
            panel_grid_major_y=p9.element_line(color="#c0bfbf"),
            panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),
            figure_size=fig_size,
            axis_text_x=p9.element_text(size=8),
            axis_text_y=p9.element_text(size=8),
            axis_title_x=p9.element_text(size=9),
            axis_title_y=p9.element_text(size=9),
            plot_title=p9.element_text(size=10, weight="bold"),
        )
    )
    return p


def plot_feature_profile(feature, X_raw, y, column_config, preproc=None,
                         fig_size=(8, 4)):
    """Dual-axis: barras (frecuencia) + linea (tasa diabetes) por
    categoria/bin, ordenado por proporcion de target=1."""  # noqa: E501
    data = pd.DataFrame({'feature': X_raw[feature].copy(), 'target': y})
    cfg = column_config.get(feature, {})
    is_binned = cfg.get('transform') == 'binning'

    if is_binned and preproc is not None and feature in preproc.binners_:
        binner = preproc.binners_[feature]
        tbl = binner.binning_table.build()
        tbl = tbl[tbl['Bin'].astype(str).str.strip() != ''].copy()
        idx_to_label = {idx: r['Bin'] for idx, r in tbl.iterrows()}
        bin_indices = binner.transform(data['feature'].values, metric='bins')
        data['group'] = [idx_to_label.get(i, f'Bin {i}') for i in bin_indices]
    else:
        data['group'] = data['feature'].astype(str)

    agg = (
        data.groupby('group', observed=True)
        .agg(count=('target', 'count'), prop_diabetes=('target', 'mean'))
        .reset_index()
    )
    agg['freq_norm'] = agg['count'] / agg['count'].sum() * 100
    agg = agg.sort_values('prop_diabetes', ascending=True).reset_index(drop=True)

    x_labels = agg['group'].tolist()
    x = np.arange(len(agg))

    fig, ax1 = plt.subplots(figsize=fig_size)
    ax1.bar(x, agg['freq_norm'].values, width=0.6, color='gray',
            alpha=0.35, edgecolor='gray', linewidth=1)
    ax1.set_ylabel('Frecuencia relativa (%)', fontsize=9)
    ax1.set_xlabel(feature, fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)

    ax2 = ax1.twinx()
    ax2.plot(x, agg['prop_diabetes'].values * 100, 'o-', color='darkred',
             linewidth=1.5, markersize=4, zorder=5)
    ax2.set_ylabel('Tasa de diabetes (%)', fontsize=9, color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')

    for i, v in enumerate(agg['prop_diabetes'].values):
        ax2.text(i, v * 100 + 1.5, f'{v*100:.1f}%', ha='center', va='bottom',
                 fontsize=7, color='darkred',
                 path_effects=[pe.Stroke(linewidth=2, foreground='white'),
                               pe.Normal()])

    ax1.set_title(feature, fontsize=10, weight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_axisbelow(True)
    fig.tight_layout()
    return fig

# Calcutation

def iv_variable (data,col,target):
    data_grouped=(
    data
    .groupby(col,dropna=False)
    .agg(Count = (f'{col}','size'),
         Non_Event=(f'{target}', lambda x: (x == 0).sum()),
         Event=(f'{target}', lambda x: (x == 1).sum())
        )
    )
    data_grouped['count_%'] = data_grouped['Count']/data_grouped['Count'].sum()
    data_grouped['event_rate'] = data_grouped['Event']/(data_grouped['Event']+data_grouped['Non_Event'])
    data_grouped['WoE'] = np.log((data_grouped['Non_Event'] / data_grouped['Non_Event'].sum()) / (data_grouped["Event"] /data_grouped["Event"].sum()))
    data_grouped['IV'] = ((data_grouped['Non_Event'] / data_grouped['Non_Event'].sum()) - (data_grouped["Event"] /data_grouped["Event"].sum()))*data_grouped['WoE']
    #return data_grouped
    # assuming your DataFrame is called df
    summary = {
        'Variable': col,
        #'Unique_values': data_grouped[col].unique().shape[0],
        "Count": data_grouped["Count"].sum(),
        "Non_Event": data_grouped["Non_Event"].sum(),
        "Event": data_grouped["Event"].sum(),
        "count_%": data_grouped["count_%"].sum(),  # should be 1.0
        "event_rate": data_grouped["Event"].sum() / data_grouped["Count"].sum(),
        "IV total": data_grouped["IV"].sum(),
    }
    return summary