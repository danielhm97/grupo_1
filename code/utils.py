import pandas as pd
import numpy as np
import plotnine as p9
from plotnine import ggplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import warnings
from plotnine.exceptions import PlotnineWarning
from scipy.stats import kruskal, levene, chi2_contingency


# Plot the histogram
def histogram_plot(data,col,color_general,color_mean,color_median,fig_size_histogram):
    mean_value = data[col].mean()
    median_value = data[col].median()

    histogram_plot = (
        p9.ggplot(data, p9.aes(x=col))
        + p9.geom_histogram(
            bins=50,
            alpha=0.2,
            fill = color_general,
            color = color_general,
            position='identity'
        )
        +p9.geom_vline(
            xintercept = mean_value,
            color = color_mean,
            linetype='dashed',
            size=0.4
        )
        +p9.geom_vline(
            xintercept = median_value,
            color = color_median,
            linetype='solid',
            size=0.4
        )
        + p9.labs(
            title=f'{col}',
            x=f'media(linea roja): {mean_value:.1f}, mediana(linea negra):{median_value:.1f}',
            y='Frecuencia'
        )        
        + p9.scale_y_continuous(expand=(0, 0))
        + p9.theme(
            panel_background=p9.element_rect(fill="#ffffff"),
            plot_background=p9.element_rect(fill='#ffffff'),
            panel_grid_major_y=p9.element_line(color="#c0bfbf"),
            panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),
            figure_size=fig_size_histogram,
            axis_text_x=p9.element_text(size=7),
            axis_text_y=p9.element_text(size=7),
            axis_title_x=p9.element_text(size=8),
            axis_title_y=p9.element_text(size=8),
            plot_title=p9.element_text(size=9, weight="bold"),
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=7),
        )
    )
    return histogram_plot

# Plot the boxplot
def box_plot(data,col,color_general,fig_size_box_plot):
    box_plot = (
        p9.ggplot(data, p9.aes(y=col)) +
        p9.geom_boxplot(outlier_alpha=0.4, 
                        alpha=0.2,
                        width=0.1,
                        fill = color_general,
                        color = color_general
                        )
        + p9.ggtitle(col)
        + p9.theme(
            panel_background=p9.element_rect(fill="#ffffff"),
            plot_background=p9.element_rect(fill='#ffffff'),
            strip_background=p9.element_rect(fill='#ffffff'),
            axis_ticks_major_y = p9.element_blank(),
            axis_ticks_major_x = p9.element_blank(),
            axis_text_x=p9.element_blank(),
            panel_grid_major_y=p9.element_line(color="#c0bfbf"),
            panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),
            axis_title_y=p9.element_blank(),
            plot_title=p9.element_text(size=9, weight="bold"),
            figure_size=fig_size_box_plot
        )                  
    )   
    return box_plot

# Describe and missing (nan) as text
def desc_plot(data,col):
    desc = data[col].describe()
    desc_text = "\n".join(f"{idx} : {val:.6g}" for idx, val in desc.items())

    col_median = data[col].median()
    desc_text = desc_text + f'\nmedian : {col_median}'
    col_nas = data[col].isna().sum()
    desc_text = desc_text + f'\nmissing (nan) : {col_nas}'
    
    return desc_text

# Plot all the cuantitative info together
def plot_var_cuantitative(histogram_plot,box_plot,desc_text,fig_size_histogram,fig_size_box_plot,fig_size_desc):
    # Silence the warning when saving into the buffer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlotnineWarning)

        # Crear divisiones relativas al ancho de cada plot
        widths = [fig_size_histogram[0],fig_size_box_plot[0],fig_size_desc[0]]
        total_width = sum(widths)
        total_height = np.array([fig_size_histogram[1],fig_size_box_plot[1],fig_size_desc[1]]).max()

        fig = plt.figure(figsize=(total_width,total_height), dpi=300)
        gs = gridspec.GridSpec(1, 3, width_ratios=widths, figure=fig)
        plots = [histogram_plot, box_plot, desc_text]

        for i, p in enumerate(plots):
            ax = fig.add_subplot(gs[i])
            buf = io.BytesIO()

            if isinstance(p, ggplot):
                p.save(buf, format='png', dpi=300)
                buf.seek(0)
                img = plt.imread(buf)
                ax.imshow(img)                   
            elif isinstance(p,str):
                ax.text(0.5, 0.5, desc_text, fontsize=10,va='center', ha='center', transform=ax.transAxes)

            ax.axis('off')

        return fig    

# Plot categorical variables
def bar_plot(data, col,color_general,fig_size,nudge_y):

    bar_data = (data[col]
                .astype(str)
                .value_counts(dropna=False)
                .reset_index(name='frecuencia')
                .sort_values('frecuencia',ascending=True)
                .reset_index(drop=True)
            )
    bar_data[col] = bar_data[col].fillna('Valor perdido (nan)')
    bar_data[col] = bar_data[col].str.strip()
    bar_data[col] = pd.Categorical(bar_data[col],categories=bar_data[col],ordered=True)
    bar_data['percentage'] = (bar_data['frecuencia']*100/(bar_data['frecuencia'].sum())).round(2)
    bar_data['label'] = bar_data['frecuencia'].astype(str) + '-' + bar_data['percentage'].astype(str)+'%'
    y_top_lim = bar_data['frecuencia'].max()

    bar_plot = (
        p9.ggplot(bar_data,p9.aes(x=col,y='frecuencia'))
        + p9.geom_col(
                      color = color_general,
                      fill = color_general,
                      alpha = 0.7
        )
        + p9.geom_text(
            p9.aes(label='label'),
            va='center',
            nudge_y=nudge_y,
            show_legend= False,
            size = 8,
            color = 'black'
        ) 
        + p9.labs(title=f"Distribución de '{col}', frecuencia y porcentaje")
        + p9.scale_y_continuous(expand=(0, 0), limits=(0,y_top_lim*1.10))
        + p9.coord_flip(xlim=None, ylim=None, expand=True)
        + p9.theme(
            panel_background=p9.element_rect(fill="#ffffff"),
            plot_background=p9.element_rect(fill='#ffffff'),
            strip_background=p9.element_rect(fill='#ffffff'),
            panel_grid_major_x=p9.element_line(color="#c0bfbf"),
            panel_grid_minor_x=p9.element_line(color="#e6e4e4ff"),
            axis_text_x=p9.element_text(size=7),
            axis_title_x=p9.element_blank(),
            axis_title_y=p9.element_blank(),
            plot_title=p9.element_text(size=12, weight="bold"),
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=7),
            figure_size=fig_size
        )
    )

    return bar_plot


def hist_per_target_plot(data,col,target_col,colors_per_group,fig_size):
    data_plot = data.copy()
    data_plot[target_col] = data_plot[target_col].astype(str)

    means = (
        data_plot
        .groupby(target_col, as_index=False)
        .agg(mean=(col, 'mean'))
    )

    mean_no_diabetes = means[means[target_col]=='0']['mean'].values[0]
    mean_diabetes = means[means[target_col]=='1']['mean'].values[0]
    
    plot = (
        p9.ggplot(data_plot, p9.aes(x=col, y='..density..',fill=target_col,color=target_col))
        + p9.geom_histogram(
            bins=50,
            alpha=0.2,
            position='identity',
            show_legend = False
        )
        + p9.geom_vline(
            data=means,
            mapping=p9.aes(xintercept='mean',color=target_col),
            linetype='dashed',
            size=0.8,
            show_legend = False
        )
        + p9.scale_y_continuous(expand=(0, 0))
        + p9.scale_fill_manual(values=colors_per_group, labels=[0,1])
        + p9.scale_color_manual(values=colors_per_group, labels=[0,1])
        + p9.labs(
            fill =target_col,
            color=target_col,
            title=f"Distribucion de '{col}', comparando grupos",
            x=f'media(diabetes:0): {mean_no_diabetes:.1f}, media(diabetes:1):{mean_diabetes:.1f}',
            y='Densidad'
        )
        + p9.theme(
            panel_background=p9.element_rect(fill="#ffffff"),
            plot_background=p9.element_rect(fill='#ffffff'),
            panel_grid_major_y=p9.element_line(color="#c0bfbf"),
            panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),
            figure_size=fig_size,
            axis_text_x=p9.element_text(size=7),
            axis_text_y=p9.element_text(size=7),
            axis_title_x=p9.element_text(size=8),
            axis_title_y=p9.element_text(size=8),
            plot_title=p9.element_text(size=9, weight="bold"),
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=7),
        )
    )
    
    return plot    

def box_per_target_plot(data,col,target_col,colors_per_group,fig_size_box_plot):
    data_plot = data.copy()
    data_plot[target_col] = data_plot[target_col].astype(str)

    box_plot = (
            p9.ggplot(data_plot, p9.aes(x=target_col ,y=col,fill=target_col,color=target_col)) +
            p9.geom_boxplot(outlier_alpha=0.4, 
                            alpha=0.2,
                            width=0.5)
            + p9.ggtitle(col)
            + p9.scale_fill_manual(values=colors_per_group, labels=[0,1])
            + p9.scale_color_manual(values=colors_per_group, labels=[0,1])
            + p9.theme(
                panel_background=p9.element_rect(fill="#ffffff"),
                plot_background=p9.element_rect(fill='#ffffff'),
                strip_background=p9.element_rect(fill='#ffffff'),
                axis_ticks_major_y = p9.element_blank(),
                axis_ticks_major_x = p9.element_blank(),
                panel_grid_major_y=p9.element_line(color="#c0bfbf"),
                panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),
                axis_title_y=p9.element_blank(),
                axis_title_x=p9.element_blank(),
                plot_title=p9.element_text(size=9, weight="bold"),
                figure_size=fig_size_box_plot
            )                  
        )
    return box_plot


def cat_variables_per_group(data,col,col_target,colors_per_group,fig_size):
    discrete_freq = data.groupby(col,dropna=False)[col_target].value_counts(normalize=True, dropna=False).reset_index()
    discrete_freq[col] = discrete_freq[col].astype(str)
    discrete_freq[col] = discrete_freq[col].fillna('Valor perdido (nan)')
    discrete_freq['proportion'] = discrete_freq['proportion']*100
    discrete_freq[col_target] = pd.Categorical(discrete_freq[col_target].astype(str),categories=['1','0'],ordered=True)
    
    plot = (
        p9.ggplot(discrete_freq, p9.aes(x=col,y='proportion',fill=col_target,color=col_target))
        + p9.geom_col(alpha = 0.2)
        + p9.scale_fill_manual(values=colors_per_group, labels=['1','0'])
        + p9.scale_color_manual(values=colors_per_group, labels=['1','0']) 
        + p9.labs(title =f"Distribución de '{col}', comparando grupos")   
        + p9.theme(
            panel_background=p9.element_rect(fill="#ffffff"),
            plot_background=p9.element_rect(fill='#ffffff'),
            panel_grid_major_y=p9.element_line(color="#c0bfbf"),
            panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),
            figure_size=fig_size,
            axis_text_x=p9.element_text(size=7),
            axis_text_y=p9.element_text(size=7),
            axis_title_x=p9.element_blank(),
            axis_title_y=p9.element_blank(),
            plot_title=p9.element_text(size=9, weight="bold"),
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=7),
        )
    )
    return plot

def cross_tab_per_target (data,col,target_col):
    '''
    target_col and col must be categoricals
    '''
    freq = pd.crosstab(data[col], data[target_col], margins=True)
    freq = freq.rename(columns={'All': 'Total Frecuencia'})

    prop = pd.crosstab(data[col], data[target_col], normalize="index", margins=True)
    prop['Total'] = prop[0] + prop[1]
    prop = (prop*100).round(2)
    cross_tab = pd.concat([freq, prop], axis=1, keys=["Frecuencia", "Porcentaje por clase"])
    cross_tab = cross_tab.rename(index = {'All':'Total'})
    return cross_tab

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


#Código V de Cramér:                                                                                                                                                                                                   import pandas as pd
import numpy as np
from scipy.stats import kruskal, levene, chi2_contingency


def _format_p(p: float) -> str:
    """Formatea el p-valor de forma legible."""
    if p == 0.0:
        return "< 0.001"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def _format_v(v: float) -> str:
    """Formatea el V de Cramér de forma legible."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    return f"{v:.3f}"


def _cramers_v_from_chi2(chi2: float, n: int, r: int, c: int) -> float:
    """
    Calcula V de Cramér a partir de chi2, n y dimensiones de la tabla.
    V = sqrt( chi2 / (n * (min(r,c) - 1)) )
    """
    k = min(r, c)
    if n <= 0 or k <= 1:
        return np.nan
    return float(np.sqrt(chi2 / (n * (k - 1))))


def desc_text_per_target(data: pd.DataFrame, col: str, target_col: str) -> str:
    """
    Devuelve un texto con los p-values apropiados para col vs target_col.
    """
    df = data[[col, target_col]].dropna()

    if df.empty:
        return "No hay datos suficientes."

    y = df[target_col]
    x = df[col]

    # Nº de valores únicos
    n_unique = x.nunique()

    # El texto donde se guardan los resultados
    texto = ""

    # ---- VARIABLE CONTINUA REAL ----
    if pd.api.types.is_numeric_dtype(x) and n_unique > 3:

        grupos = [x[y == k] for k in sorted(y.unique())]

        # Kruskal-Wallis
        _, p_kw = kruskal(*grupos)

        # Levene (igualdad de varianzas)
        _, p_lev = levene(*grupos, center="median")

        texto += f"Kruskal-Wallis p-value: {_format_p(p_kw)}\n"
        texto += f"Levene p-value: {_format_p(p_lev)}"

        return texto

    # ---- VARIABLE CATEGÓRICA / BINARIA ----
    else:
        ct = pd.crosstab(x, y)

        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return texto + "No se puede aplicar test."

        # Chi-cuadrado
        chi2, p_chi, dof, expected = chi2_contingency(ct)

        # V de Cramér (tamaño del efecto)
        n = int(ct.values.sum())
        r, c = ct.shape
        v_cramer = _cramers_v_from_chi2(chi2=chi2, n=n, r=r, c=c)

        texto += f"Chi-cuadrado p-value: {_format_p(p_chi)}\n"
        texto += f"V de Cramér: {_format_v(v_cramer)}"

        return texto




# Plot corr matrix
def corr_plot(data, numeric_var,fig_size,corr_filter=0):
    # Calcular matriz de correlación
    corr_matrix = data[numeric_var].corr()
    col_order = corr_matrix.columns.tolist()

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
        + p9.labs(title="Correlation Matrix", fill="Corr")
    )

    return corr_plot

## Return a df with the corr_matrix just for the couples of variables that pass the filter and are not the same eg.(gdp vs gdp)
#def corr_matrix(data, numeric_var,corr_filter = 0):
#    corr_matrix = data[numeric_var].corr()
#    col_order = corr_matrix.columns.tolist()
#
#    # Creamos una matriz 'mask' para poder quedarnos solo con el triangulo inferior
#    mask = np.zeros_like(corr_matrix, dtype=bool)
#    # Nos quedamos con el triangulo inferior y la diagonal
#    mask[np.triu_indices_from(mask, k=1)] = True
#
#    # Utilizamos mask para filtrar y formateamos la matriz para plotnine
#    corr_matrix = corr_matrix.mask(mask).stack().reset_index(name='value')
#    corr_matrix.columns = ['var1', 'var2', 'value']
#
#    corr_matrix['var1'] = pd.Categorical(corr_matrix['var1'], categories=col_order)
#    corr_matrix['var2'] = pd.Categorical(corr_matrix['var2'], categories=col_order)
#    corr_matrix = corr_matrix[(np.abs(corr_matrix['value'])>=corr_filter) & (corr_matrix['var1']!=corr_matrix['var2'])]
#    corr_matrix['var1'] = corr_matrix['var1'].cat.remove_unused_categories()
#    corr_matrix['var2'] = corr_matrix['var2'].cat.remove_unused_categories()
#    corr_matrix = corr_matrix.reset_index(drop = True)
#    return corr_matrix
#
## Return an scatter_plot
#def scatter_plot_corr (data,corr_matrix,i,fig_size):
#    x_values = corr_matrix['var1'][i]
#    y_values = corr_matrix['var2'][i]
#    corr_value = corr_matrix['value'][i]
#    scatter_plot = (
#        p9.ggplot(data, p9.aes(x=x_values, y=y_values))
#        + p9.geom_point(alpha=0.6, size=2)  # alpha helps with overlapping points
#        + p9.geom_smooth(method='lm', color='blue')  # Adds a linear trend line
#        + p9.labs(
#            title=f'{x_values} vs {y_values}, r={corr_value:.3f}',
#            x=f'{x_values}, n={data.shape[0]}'
#
#        )
#        + p9.theme(
#            panel_background=p9.element_rect(fill="#ffffff"),
#            plot_background=p9.element_rect(fill='#ffffff'),
#            strip_background=p9.element_rect(fill='#ffffff'),
#            panel_grid_major_x=p9.element_line(color="#c0bfbf"),
#            panel_grid_minor_x=p9.element_line(color="#e6e4e4ff"),
#            panel_grid_major_y=p9.element_line(color="#c0bfbf"),
#            panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),        
#            axis_text_x=p9.element_text(size=7),
#            axis_text_y=p9.element_text(size=7),
#            axis_title_x=p9.element_text(size=8),
#            axis_title_y=p9.element_text(size=8),
#            plot_title=p9.element_text(size=8),
#            figure_size=fig_size
#        )
#
#    )
#    return scatter_plot
#
## Plot in a grid all the plot that are in an array
#def plot_grid(plots,fig_size_plot,grid_rows,grid_cols):
#    # Silence the warning when saving into the buffer
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore", PlotnineWarning)
#        fig = plt.figure(figsize=fig_size_plot, dpi=300)
#        gs = gridspec.GridSpec(grid_rows, grid_cols, figure=fig)
#
#        for i, p in enumerate(plots):
#            ax = fig.add_subplot(gs[i])
#            buf = io.BytesIO()
#
#            if isinstance(p, ggplot):
#                p.save(buf, format='png', dpi=300)
#                buf.seek(0)
#                img = plt.imread(buf)
#                ax.imshow(img)
#                ax.axis('off')
#
#    return fig    
#    
## Plot adjusted r2 evolution
#def plot_r2_evolution(info_iterations,fig_size,dpi):
#    fig, ax1 = plt.subplots(figsize=fig_size,dpi=dpi)
#
#    # --- Left axis: Adj R² ---
#    color_line = 'steelblue'
#    ax1.set_xlabel('Número Features', fontsize=9)
#    ax1.set_ylabel('R² ajustado', color='black', fontsize=9)
#    ax1.plot(info_iterations['n_features'], info_iterations['adj_r2'],
#             color=color_line, marker='o', linestyle='--', linewidth=2)
#    ax1.tick_params(axis='y', labelcolor='black',labelsize=6)
#    ax1.tick_params(axis='x', labelsize=6)
#
#    # Add labels on the dots
#    for x, y in zip(info_iterations['n_features'], info_iterations['adj_r2']):
#        ax1.text(x, y + 0.006, f'{y:.3f}', ha='center', va='bottom', color='black', fontsize=6)
#
#    # --- Right axis: % improvement ---
#    ax2 = ax1.twinx()  # create a second y-axis
#    color_bar = 'seagreen'
#    ax2.set_ylabel('% Mejora', color='black')
#    ax2.bar(info_iterations['n_features'], info_iterations['pct_improvement'],
#            color=color_bar, alpha=0.5, width=0.5)
#    ax2.set_yticks([])
#    ax2.set_ylim(0, 20) 
#
#    # Remove top and right spines for both axes
#    ax1.spines['top'].set_visible(False)
#    ax1.spines['right'].set_visible(False)
#    ax2.spines['top'].set_visible(False)
#    ax2.spines['right'].set_visible(False)
#
#    # Reduce thickness of bottom and left spines
#    ax1.spines['bottom'].set_linewidth(0.5)
#    ax1.spines['left'].set_linewidth(0.5)
#
#    # Optional: add grid
#    ax1.grid(True, linestyle='-', alpha=0.2, axis='y')
#
#    # Optional: add values on top of bars
#    for x, y in zip(info_iterations['n_features'], info_iterations['pct_improvement']):
#        if y>0:
#            ax2.text(x, y + 0.15, f'{y:.2f}%', ha='center', color='black', fontsize=7)
#    
#    plt.title('Evolución de R² ajustado y % de mejora por número de features', fontsize=10, pad=18)
#    return fig
#
## Best subset selection 
#def best_subset_selection(X, y, max_features=None, metric='adj_r2', method='best'):
#    if max_features is None:
#        max_features = len(X.columns)
#    n = len(y)
#
#    def get_metrics(X_subset, y):
#        X_sm = sm.add_constant(X_subset)
#        model_sm = sm.OLS(y, X_sm).fit()
#        y_pred = model_sm.predict(X_sm)
#        mse = mean_squared_error(y, y_pred)
#        r2 = r2_score(y, y_pred)
#        p = X_subset.shape[1]
#        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
#        aic = model_sm.aic
#        bic = model_sm.bic
#        return mse, r2, adj_r2, aic, bic, model_sm
#
#    best_results = []
#    resultados = []
#    if method == 'best':
#        for k in range(1, max_features + 1):
#            k_results = []
#            for combo in combinations(X.columns, k):
#                X_subset = X[list(combo)]
#                mse, r2, adj_r2, aic, bic, model_sm = get_metrics(X_subset, y)
#                k_results.append({'features': combo, 'mse': mse, 'r2': r2, 'adj_r2': adj_r2,
#                                  'aic': aic, 'bic': bic, 'model': model_sm})
#                
#            k_results = sorted(k_results, key=lambda x: x[metric], reverse=True)
#            best_results.append(k_results[0])
#            resultados = resultados + k_results
#
#                
#    else:
#        raise ValueError('Método no reconocido')
#
#    # Ordenar el resultado final
#    reverse_order = metric in ['r2', 'adj_r2']
#    resultados = sorted(resultados, key=lambda x: x[metric], reverse=reverse_order)
#    best_results = sorted(best_results, key=lambda x: x[metric], reverse=reverse_order)
#    return resultados, best_results
#
#
#def scatter_plot_th(data,x_values,y_values,y_intercept,fig_size,title,x_label,y_label):
#    scatter_plot = (p9.ggplot(data, p9.aes(x=x_values, y=y_values))
#            + p9.geom_point(alpha=0.6, size=2)  # alpha helps with overlapping points                        
#            + p9.geom_hline(
#                p9.aes(yintercept=y_intercept),
#                color='red',
#                linetype='dashed',
#                size=0.8,
#                alpha=0.8
#            )
#            + p9.labs(
#                title=title,
#                x=x_label,
#                y=y_label
#            )
#            #+ p9.scale_y_continuous(
#            #    breaks=np.arange(0, max(data[y_values]), 0.02)
#            #)
#            + p9.theme(
#                panel_background=p9.element_rect(fill="#ffffff"),
#                plot_background=p9.element_rect(fill='#ffffff'),
#                strip_background=p9.element_rect(fill='#ffffff'),
#                panel_grid_major_x=p9.element_blank(),
#                panel_grid_minor_x=p9.element_blank(),
#                panel_grid_major_y=p9.element_line(color="#c0bfbf"),
#                panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),        
#                axis_text_x=p9.element_text(size=7),
#                axis_text_y=p9.element_text(size=7),
#                axis_title_x=p9.element_text(size=8),
#                axis_title_y=p9.element_text(size=8),
#                plot_title=p9.element_text(size=10),
#                figure_size=fig_size
#            )    
#    )
#    return scatter_plot
#
#def qq_plot (data,col,color_general,fig_size):
#    qq_plot = (
#        p9.ggplot(data, p9.aes(sample=col))
#        + p9.stat_qq(alpha = 0.4, color = color_general)
#        + p9.stat_qq_line(color = 'red')
#        + p9.labs(
#            title='Q-Q Plot residuos',
#            x='Cuantiles teóricos',
#            y='Cuantiles de la muestra'
#        )
#        + p9.theme(
#            panel_background=p9.element_rect(fill="#ffffff"),
#            plot_background=p9.element_rect(fill='#ffffff'),
#            panel_grid_major_y=p9.element_line(color="#c0bfbf"),
#            panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),
#            figure_size=fig_size,
#            axis_text_x=p9.element_text(size=7),
#            axis_text_y=p9.element_text(size=7),
#            axis_title_x=p9.element_text(size=8),
#            axis_title_y=p9.element_text(size=8),
#            plot_title=p9.element_text(size=9, weight="bold"),
#            legend_title=p9.element_text(size=8),
#            legend_text=p9.element_text(size=7),
#        ) 
#    )
#    return qq_plot
#
#def hist_density_plot(data,col,color_general,fig_size):
#    hist_density_plot = (
#        p9.ggplot(data, p9.aes(x=col))
#        + p9.geom_histogram(
#            p9.aes(y='..density..'),
#            bins=30,
#            alpha=0.2,
#            fill = color_general,
#            color = color_general,
#            position='identity'
#        )
#        + p9.geom_density(
#            bw='nrd0',
#            color='black',
#            size=0.25
#        )
#        + p9.labs(
#            title='Histograma de residuos',
#            y='Densidad'
#        )
#        + p9.theme(
#            panel_background=p9.element_rect(fill="#ffffff"),
#            plot_background=p9.element_rect(fill='#ffffff'),
#            panel_grid_major_y=p9.element_line(color="#c0bfbf"),
#            panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),
#            figure_size=fig_size,
#            axis_text_x=p9.element_text(size=7),
#            axis_text_y=p9.element_text(size=7),
#            axis_title_x=p9.element_blank(),
#            axis_title_y=p9.element_text(size=8),
#            plot_title=p9.element_text(size=9, weight="bold"),
#            legend_title=p9.element_text(size=8),
#            legend_text=p9.element_text(size=7),
#        )        
#    )
#    return hist_density_plot
#
#