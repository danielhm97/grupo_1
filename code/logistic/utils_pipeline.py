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
import numpy as np
import pandas as pd
import plotnine as p9


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import warnings
from plotnine.exceptions import PlotnineWarning


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

def plot_confusion_matrix(cm,size, title=None):
    """
    Confusion matrix heatmap using plotnine.
    Shows counts and row percentages.
    """
    n = cm.shape[0]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0)

    data = []
    for i in range(n):
        for j in range(n):
            label_x =  "No Diabético" if i == 0 else "Diabético"
            label_y = "No Diabético" if j == 0 else "Diabético"
            data.append({
                "Actual": label_x,
                "Predicted": label_y,
                "Count": cm[i, j],
                "Pct": cm_pct[i, j],
                "Diagonal": "Correct" if i == j else "Incorrect",
                "Label": f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"
            })

    df = pd.DataFrame(data)

    # Map pct to alpha: scale within [0.15, 0.9] so even 0% tiles are visible
    df["Alpha"] = 0.1 + (df["Pct"] / 100) * 0.9

    # Build per-row fill color using base colors + alpha baked in
    import matplotlib.colors as mcolors

    def make_color(row):
        base = "seagreen" if row["Diagonal"] == "Correct" else "red"
        r, g, b = mcolors.to_rgb(base)
        # Blend with white based on alpha
        a = row["Alpha"]
        return f"#{int((r*a + 1*(1-a))*255):02x}{int((g*a + 1*(1-a))*255):02x}{int((b*a + 1*(1-a))*255):02x}"

    df["FillColor"] = df.apply(make_color, axis=1)


    df["TileID"] = df.index.astype(str)
    fill_values = dict(zip(df["TileID"], df["FillColor"]))

    plot = (
        p9.ggplot(df, p9.aes(x="Predicted", y="Actual"))
        + p9.geom_tile(
            p9.aes(fill="TileID"),
            color="black",
        )
        + p9.geom_text(
            p9.aes(label="Label"),
            size=8,
            fontweight="bold"
        )
        + p9.scale_fill_manual(values=fill_values)
        + p9.labs(title=title, x='Predicho', y='Actual', fill="")
        + p9.theme(
            panel_background=p9.element_rect(fill="#ffffff"),
            plot_background=p9.element_rect(fill='#ffffff'),
            panel_grid_major_y=p9.element_blank(),
            panel_grid_minor_y=p9.element_blank(),
            panel_grid_major_x=p9.element_blank(),
            panel_grid_minor_x=p9.element_blank(),
            axis_ticks_major_x=p9.element_blank(),
            axis_ticks_major_y=p9.element_blank(),
            figure_size=size,
            #axis_title_y=p9.element_text(margin={"r": 0}),
            axis_text_x=p9.element_text(size=9, color='black',margin={"t": -2}),
            axis_text_y=p9.element_text(size=9, color='black', rotation=90, va='center',
                                        margin={"r": -8}),
            plot_title=p9.element_text(size=9, weight="bold", margin={"t": 8, "b": 8}),
            legend_position="none",
        )
    )

    return plot

import plotnine as p9
from sklearn.metrics import precision_recall_curve, average_precision_score

def auc_box(X_train, y_train, X_test, y_test, model_pipe):
    aucs = {}
    for name, X, y in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        y_proba = model_pipe.predict_proba(X)[:, 1]
        aucs[name] = average_precision_score(y, y_proba)

    auc_label = f"Train AUC: {aucs['Train']:.3f}\nTest  AUC: {aucs['Test']:.3f}"

    return auc_label

def df_pr_curve(X_train, y_train, X_test, y_test, model_pipe):
    # Build dataframes for each curve
    df_plot = []
    for name, X, y in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        y_proba = model_pipe.predict_proba(X)[:, 1]
        pr, re, _ = precision_recall_curve(y, y_proba)
        df_pr = pd.DataFrame({'Precision': pr, 'Recall': re})
        df_pr['Model'] = name
        df_plot.append(df_pr)

    df_plot = pd.concat(df_plot, ignore_index=True)
    df_plot['Model'] = pd.Categorical(df_plot['Model'], categories=['Train', 'Test'], ordered=True)
    return df_plot

def pr_curve_plot(df_plot,
                  random_level, 
                  auc_label,
                  title = 'Curva Precision-Recall',
                  size = (6,5)
                  ):
    color_map = {'Train': '#a00000', 'Test': '#1a80bb'}

    plot = (
        p9.ggplot(df_plot, p9.aes(x='Recall', y='Precision', color='Model'))
        + p9.geom_line(size=1, alpha=0.7)
        + p9.geom_hline(yintercept=random_level, linetype='dashed', color='grey')
        + p9.annotate('text', x=0.25, y=random_level + 0.02,
                   label=f'Random ({random_level:.3f})', color='black', ha='right', size=9
                )
        + p9.annotate(
            'label',
            x=0.97, y=0.97,
            label=auc_label,
            ha='right', va='top',
            size=9,
            color='black',
            fill='white',
            label_padding=0.4
        )
        + p9.scale_color_manual(values=color_map)
        + p9.scale_y_continuous(limits=(0, 1.0), expand=(0.05, 0), breaks=np.linspace(0, 1.0, 6))
        + p9.scale_x_continuous(limits=(0, 1.0), expand=(0.05, 0), breaks=np.linspace(0, 1.0, 6))
        + p9.labs(title=title, x='Recall', y='Precision', color='')
        + p9.theme(
            panel_background=p9.element_rect(fill="#ffffff"),
            plot_background=p9.element_rect(fill='#ffffff'),
            panel_grid_major_y=p9.element_line(color="#c0bfbf"),
            panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),
            panel_grid_major_x=p9.element_line(color="#c0bfbf"),
            panel_grid_minor_x=p9.element_line(color="#e6e4e4ff"),
            figure_size=size,
            axis_text_x=p9.element_text(size=9,color='black'),
            axis_text_y=p9.element_text(size=9,color='black'),
            plot_title=p9.element_text(size=10,margin={"t": 8, "b": 8},weight="bold"),
            legend_position="right"
        )
    )

    return plot



def plot_feature_profile(feature, data_base, target, fig_size=(8, 4)):
    """Dual-axis: barras (frecuencia) + linea (tasa target) por
    categoria, ordenado por proporcion de target=1."""  # noqa: E501
    data = pd.DataFrame({'feature': data_base[feature].copy(), 'target': target})
    data['group'] = data['feature'].astype(str)

    agg = (
        data.groupby('group', observed=True)
        .agg(count=('target', 'count'), prop_diabetes=('target', 'mean'))
        .reset_index()
    )
    agg['freq_norm'] = agg['count'] / agg['count'].sum() * 100
    agg = agg.sort_values('prop_diabetes', ascending=True).reset_index(drop=True)

    x_labels = agg['group'].tolist()

    if x_labels == ['0', '1']:
        x_labels = ['No', 'Si']
        
    x = np.arange(len(agg))

    fig, ax1 = plt.subplots(figsize=fig_size, dpi=200)
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    ax1.bar(x, 
            agg['freq_norm'].values, 
            width=0.6, 
            color='steelblue',
            alpha=0.7, 
            edgecolor='steelblue', 
            linewidth=1
        )

    ax1.set_ylabel('Frecuencia relativa (%)', fontsize=9)
    ax1.set_xticks(x)

    if len(x_labels) > 5:
        ax1.set_xticklabels(x_labels, rotation=45, ha='left', fontsize=7)
    else:
        ax1.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=9)

    ax1.spines[['top', 'right']].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(x, agg['prop_diabetes'].values * 100, 'o-', color='red',
             linewidth=1.5, markersize=4, zorder=5)
    #ax2.set_ylabel('Tasa de diabetes (%)', fontsize=9, color='red')
    #ax2.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', right=False, labelright=False)
    ax2.spines[['top', 'right']].set_visible(False)

    for i, v in enumerate(agg['prop_diabetes'].values):
        ax2.annotate(f'{v*100:.1f}%',
                     xy=(i, v * 100),
                     xytext=(0, 8),
                     textcoords='offset points',
                     ha='center', va='bottom',
                     fontsize=7, color='black')

    y_lo, y_hi = ax2.get_ylim()
    ax2.set_ylim(y_lo, y_hi * 1.05)

    ax1.set_title(f'Frecuencia relativa vs Tasa de diabetes\n{feature}',
                  fontsize=10, 
                  #weight='bold'
                )
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
        #"count_%": data_grouped["count_%"].sum(),  # should be 1.0
        "event_rate": data_grouped["Event"].sum() / data_grouped["Count"].sum(),
        "IV total": data_grouped["IV"].sum(),
    }
    return summary



def join_plots(plots,sizes):
    # Silence the warning when saving into the buffer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlotnineWarning)

        # Crear divisiones relativas al ancho de cada plot
        widths = [s[0] for s in sizes]
        total_width = sum(widths)
        total_height = np.array([s[1] for s in sizes]).max()

        fig = plt.figure(figsize=(total_width,total_height), dpi=300)
        gs = gridspec.GridSpec(1, len(plots), width_ratios=widths, figure=fig)

        for i, p in enumerate(plots):
            ax = fig.add_subplot(gs[i])
            buf = io.BytesIO()

            if isinstance(p, ggplot):
                p.save(buf, format='png', dpi=300)
                buf.seek(0)
                img = plt.imread(buf)
                ax.imshow(img)                   
            elif isinstance(p,str):
                ax.text(0.5, 0.5, p, fontsize=10,va='center', ha='center', transform=ax.transAxes)

            ax.axis('off')

        return fig    
    

# Plot corr matrix
def corr_plot(data, numeric_var,fig_size,corr_filter=0, title=None):
    # Calcular matriz de correlación
    corr_matrix = data[numeric_var].corr()
    col_order = corr_matrix.columns.tolist()

    if title is None:
        title = "Correlation Matrix"

    # Creamos una matriz 'mask' para poder quedarnos solo con el triangulo inferior
    mask = np.zeros_like(corr_matrix, dtype=bool)
    # Nos quedamos con el triangulo inferior y la diagonal
    mask[np.triu_indices_from(mask, k=0)] = True

    # Utilizamos mask para filtrar y formateamos la matriz para plotnine
    corr_matrix = corr_matrix.mask(mask).stack().reset_index(name='value')
    corr_matrix.columns = ['var1', 'var2', 'value']

    corr_matrix['var1'] = pd.Categorical(corr_matrix['var1'], categories=col_order)
    corr_matrix['var2'] = pd.Categorical(corr_matrix['var2'], categories=col_order)
    corr_matrix = corr_matrix[np.abs(corr_matrix['value'])>=corr_filter]

    corr_plot = (
        p9.ggplot(corr_matrix, p9.aes(x='var1', y='var2', fill='value'))
        + p9.geom_tile()  # This creates the squares
        + p9.geom_text(p9.aes(label='value.round(2)'), size=8) # Add coefficients
        + p9.scale_fill_gradient2(
            low='#d7191c', 
            mid='#ffffbf', 
            high="#05b402", 
            midpoint=0, 
            limits=[-1, 1]
        )
        + p9.theme_minimal()
        + p9.theme(
            axis_text_x=p9.element_text(rotation=45, hjust=1),
            axis_title=p9.element_blank(),
            figure_size=fig_size
        )
        + p9.labs(title=title, fill="Corr")
    )

    return corr_plot


    
