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
import matplotlib.patheffects as pe


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

    col_nas = data[col].isna().sum()
    desc_text = desc_text + f'\nmissing (nan) : {col_nas}'
    
    return desc_text


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

def cat_per_group_plot(data,col,col_target,colors_per_group,fig_size,
                       order='proportion',legend_pos='right',ticks_x_rot=False,
                       alt_title=None
                    ):
    summary_df = pd.crosstab(data[col].fillna('Valor perdido (na)'),
                             data[col_target],
                             normalize="index"
                            )*100
    summary_df = summary_df.reset_index()
    summary_df = summary_df.sort_values(1,ascending=True).reset_index(drop=True)

    cat_order = summary_df[col].to_list()

    summary_df = summary_df.melt(id_vars=col,value_vars=[0,1],value_name='proportion')
    summary_df[col_target] = pd.Categorical(summary_df[col_target].astype(str),categories=['1','0'],ordered=True)

    if order=='proportion':        
        summary_df[col] = pd.Categorical(summary_df[col],categories=cat_order,ordered=True)
        
    elif order=='categories': 
        summary_df = summary_df.sort_values(col, ascending=True)
        summary_df[col] = pd.Categorical(summary_df[col],
                                         categories=summary_df[col].unique(),
                                         ordered=False)

    if ticks_x_rot==True:
        ticks_position_x = p9.element_text(size=7,rotation=45, ha='right')
    else:
        ticks_position_x = p9.element_text(size=7)
            
    if alt_title:
        plot_title = alt_title
    else:
        plot_title = col

    plot = (
        p9.ggplot(summary_df, p9.aes(x=col,y='proportion',fill=col_target,color=col_target))
        + p9.geom_col(alpha = 0.2)
        + p9.scale_fill_manual(values=colors_per_group, labels=['1','0'])
        + p9.scale_color_manual(values=colors_per_group, labels=['1','0']) 
        + p9.labs(title =f"'{plot_title}'")   
        + p9.theme(
            panel_background=p9.element_rect(fill="#ffffff"),
            plot_background=p9.element_rect(fill='#ffffff'),
            panel_grid_major_y=p9.element_line(color="#c0bfbf"),
            panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),
            figure_size=fig_size,
            axis_text_x=ticks_position_x,
            axis_text_y=p9.element_text(size=7),
            axis_title_x=p9.element_blank(),
            axis_title_y=p9.element_blank(),
            plot_title=p9.element_text(size=9, weight="bold"),
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=7),
            legend_position=legend_pos
        )
    )
    return plot

def cat_plot_num_unbinned(
    data, x_col, target_col,
    colors_per_group={'0':'seagreen','1':'red'},
    step=1.0, tick_every=None,
    fig_size=(4,4), order='categories',
    legend_pos='none', alt_title=None,
    text_size=7
):
    bcol = f"{x_col}__b"
    df = data.assign(**{bcol: (data[x_col]/step).round().astype('Int64')*step})

    p = cat_per_group_plot(
        df, bcol, target_col,
        colors_per_group=colors_per_group,
        fig_size=fig_size,
        order=order,
        legend_pos=legend_pos,
        ticks_x_rot=True,
        alt_title=alt_title
    )

    if tick_every is None:
        tick_every = 10 * step

    vals = sorted(df[bcol].dropna().astype(float).unique())
    if not vals:
        return p

    # seleccionar valores que son múltiplos de tick_every (robusto con floats)
    breaks = [v for v in vals if abs((v/tick_every) - round(v/tick_every)) < 1e-9]

    def fmt(v):
        return str(int(v)) if float(v).is_integer() else str(v)

    return (
        p
        + p9.scale_x_discrete(breaks=breaks, labels=[fmt(v) for v in breaks])
        + p9.theme(axis_text_x=p9.element_text(rotation=0, ha='center', size=text_size))
    )

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

    cross_tab_total = cross_tab.loc[['Total']]

    cross_tab = cross_tab.drop('Total')
    cross_tab = cross_tab.sort_values(('Porcentaje por clase',1), ascending=True)
    cross_tab = pd.concat([cross_tab,cross_tab_total])

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
    
def _plot_stacked_percent_ax(
    ax, data, group_col, target_col, title=None,
    order=None, colors=None, min_show=2.0, fontsize=12,
    width=0.9, alpha=0.35, edge_lw=2,
    edgecolor0='darkgreen', edgecolor1='darkred',
    ylim_top=110, grid_alpha=0.3, xtick_fontsize=10
):
    if colors is None:
        colors = {'0': 'seagreen', '1': 'red'}

    # % de clase 1 por grupo (asumiendo target binario 0/1)
    pct1 = data.groupby(group_col)[target_col].mean() * 100

    # reordenar y asegurar todas las categorías
    if order is not None:
        pct1 = pct1.reindex(order)

    # categorías sin datos -> 0
    pct1 = pct1.fillna(0.0)
    pct0 = 100 - pct1

    x = np.arange(len(pct1.index))

    # barras apiladas
    b0 = ax.bar(
        x, pct0.values, width=width,
        color=colors['0'], alpha=alpha,
        edgecolor=edgecolor0, linewidth=edge_lw
    )
    b1 = ax.bar(
        x, pct1.values, width=width, bottom=pct0.values,
        color=colors['1'], alpha=alpha,
        edgecolor=edgecolor1, linewidth=edge_lw
    )

    # etiquetas %
    def _add_labels(bars, values, bottoms=None, color=None):
        for i, rect in enumerate(bars):
            h = float(values[i])
            if h < min_show:
                continue
            y0 = 0.0 if bottoms is None else float(bottoms[i])
            y = y0 + h / 2.0

            label = f"{h:.1f}%" if h < 10 else f"{h:.0f}%"
            ax.text(
                rect.get_x() + rect.get_width() / 2.0, y, label,
                ha="center", va="center",
                fontsize=fontsize, color=color, zorder=10,
                path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()]
            )

    _add_labels(b0, pct0.values, bottoms=None, color=colors['0'])
    _add_labels(b1, pct1.values, bottoms=pct0.values, color=colors['1'])

    # ejes y estilo
    ax.set_title(f"'{title or target_col}'")
    ax.set_xticks(x)
    ax.set_xticklabels(pct1.index, fontsize=xtick_fontsize)
    ax.set_ylim(0, ylim_top)
    ax.grid(axis='y', alpha=grid_alpha)


def plot_stacked_percent_panels(
    data, group_col, target_cols,
    order=None, colors=None,
    min_show=2.0, fontsize=12,
    figsize=(21, 7), sharey=True
):
    if colors is None:
        colors = {'0': 'seagreen', '1': 'red'}

    n = len(target_cols)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=sharey)
    if n == 1:
        axes = [axes]

    for ax, t in zip(axes, target_cols):
        _plot_stacked_percent_ax(
            ax=ax,
            data=data,
            group_col=group_col,
            target_col=t,
            title=t,
            order=order,
            colors=colors,
            min_show=min_show,
            fontsize=fontsize
        )

    plt.tight_layout()
    return fig, axes

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

def scatter_plot(data,x_values,y_values,fig_size,title,x_label,y_label,colors_per_group=None,col_target=None):
    data_plot = data.copy()
    if col_target:
        data_plot[col_target] = pd.Categorical(data_plot[col_target].astype(str),categories=['0','1'],ordered=True)
        aes_scatter = p9.aes(x=x_values, y=y_values,fill=col_target,color=col_target)
    else:
        aes_scatter = p9.aes(x=x_values, y=y_values)

    scatter_plot = (p9.ggplot(data_plot,aes_scatter)
            + p9.geom_point(alpha=0.4, size=2)  # alpha helps with overlapping points
            + p9.labs(
                title=title,
                x=x_label,
                y=y_label
            )
            + p9.scale_fill_manual(values=colors_per_group)
            + p9.scale_color_manual(values=colors_per_group) 
            + p9.theme(
                panel_background=p9.element_rect(fill="#ffffff"),
                plot_background=p9.element_rect(fill='#ffffff'),
                strip_background=p9.element_rect(fill='#ffffff'),
                panel_grid_major_x=p9.element_blank(),
                panel_grid_minor_x=p9.element_blank(),
                panel_grid_major_y=p9.element_line(color="#c0bfbf"),
                panel_grid_minor_y=p9.element_line(color="#e6e4e4ff"),        
                axis_text_x=p9.element_text(size=7),
                axis_text_y=p9.element_text(size=7),
                axis_title_x=p9.element_text(size=8),
                axis_title_y=p9.element_text(size=8),
                plot_title=p9.element_text(size=10),
                figure_size=fig_size
            )    
    )
    return scatter_plot

def hist_comparative(data,col,target_col,colors_per_group,fig_size,order_categories=None):
    data_plot = data.copy()

    if order_categories:
        data_plot[target_col] = pd.Categorical(data_plot[target_col],
                                               categories=order_categories,
                                               ordered=True)
    else:
        data_plot[target_col]=data_plot[target_col].astype(str)

    means = (
        data_plot
        .groupby(target_col, as_index=False)
        .agg(mean=(col, 'mean'))
    )

    plot = (
        p9.ggplot(data_plot, p9.aes(x=col,fill=target_col))
        + p9.geom_histogram(
            bins=50,
            alpha=0.5,
            position='identity',
            color = 'black',
            size = 0.1
        )
        + p9.geom_vline(
            data=means,
            mapping=p9.aes(xintercept='mean',color=target_col),
            linetype='dashed',
            size=0.8,
            show_legend = False
        )
        + p9.scale_y_continuous(expand=(0, 0))
        + p9.scale_fill_manual(values=colors_per_group)
        + p9.scale_color_manual(values=colors_per_group)
        + p9.labs(
            fill =target_col,
            color=target_col,
            title=f"Distribución de Edad según antecedentes de tabaquismo",
            x='Edad',
            y='Cantidad de pacientes'
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

    means = means.set_index(target_col)
    means = means.round(2).T
    return plot,means  

def kruskal_table(groups: dict, continuous_vars: list) -> pd.DataFrame:
    """
    groups: dict {nombre_grupo: dataframe}
    continuous_vars: lista de variables continuas
    """
    rows = []
    for var in continuous_vars:
        samples = [g[var].dropna() for g in groups.values()]
        stat, p = kruskal(*samples)
        rows.append({
            'Variable': var,
            'Test': 'Kruskal-Wallis',
            'p-value': p
        })
    return pd.DataFrame(rows)

def cramers_v_from_ct(ct: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def chi2_binary_table(data: pd.DataFrame, group_col: str,
                      groups: list, binary_vars: list) -> pd.DataFrame:
    rows = []
    for var in binary_vars:
        ct = pd.crosstab(data[group_col], data[var]).loc[groups]
        chi2, p, _, _ = chi2_contingency(ct)
        v = cramers_v_from_ct(ct)
        rows.append({
            'Variable': var,
            'Test': 'Chi-cuadrado',
            'p-value': p,
            'V de Cramér': v
        })
    return pd.DataFrame(rows)

def proportion_table(data: pd.DataFrame, group_col: str,
                     groups: list, binary_vars: list) -> pd.DataFrame:
    return (
        data[data[group_col].isin(groups)]
        .groupby(group_col)[binary_vars]
        .mean()
        * 100
    ).round(2)