import pandas as pd
import numpy as np
import plotnine as p9
# Given the y_min and y_max, compute the coordinates for the y axis, for n labels
def y_pos_label(y_min,y_max,n_labels):
    y_pos = []
    for i in range(1,n_labels+1):
        y_pos.append(y_max-((y_max-y_min)/100)*(2*i))
    return y_pos

# Given 2 means calculate positions, placing the minor label on the left and the mayor on the right
def x_pos_label(x_min,x_max,means):
    x_pos = []
    factor = 10
    for mean in means:
        if mean == max(means):
            x_pos.append(mean+((x_max-x_min)/100)*(factor))
        else:
            x_pos.append(mean-((x_max-x_min)/100)*(factor)) 
    
    return x_pos


def get_y_max_plot(data,col,target_col):
    y_compare=[]
    y_compare.append(np.histogram(data[col],bins=50,density=True)[0].max())
    for value in data[target_col].unique():
        y_compare.append(np.histogram(data[data[target_col]==value][col],bins=50,density=True)[0].max())

    return max(y_compare)

def plot_per_group_data(data,col,target_col,colors_per_group,fig_size,y_max_plot):
    means = (
        data
        .groupby(target_col, as_index=False)
        .agg(mean=(col, 'mean'))
    )


    means['y_pos_label'] = y_max_plot-((y_max_plot-0)/100)*5
    means['x_pos_label'] = x_pos_label(data[col].min(),data[col].max(),means['mean'])
    means['label'] = means['mean'].round(1).astype(str).radd('Mean: ')

    plot = (
        p9.ggplot(data, p9.aes(x=col, y='..density..',fill=target_col,color=target_col))
        + p9.geom_histogram(
            bins=50,
            alpha=0.2,
            position='identity'
        )
        + p9.geom_vline(
            data=means,
            mapping=p9.aes(xintercept='mean', color=target_col),
            linetype='dashed',
            size=0.8
        ) 
        + p9.geom_text(
            data=means,
            mapping=p9.aes(
                x='x_pos_label',
                y='y_pos_label',
                label='label',
                color=target_col
            ),
            ha='center',
            size=5,
            show_legend = False
        )
        + p9.scale_y_continuous(expand=(0, 0), limits=(0,y_max_plot))
        + p9.scale_fill_manual(values=colors_per_group, labels=['True','False'])
        + p9.scale_color_manual(values=colors_per_group, labels=['True','False'])
        + p9.labs(
            fill =target_col,
            color=target_col,
            title=f"Distribucion de '{col}', comparando grupos",
            x=col,
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

def plot_all_data(data,col,color_general,fig_size,y_max_plot):
    mean_value = data[col].mean()
    mean_label = f'Mean: {mean_value:.1f}'
    y_max_plot = y_max_plot
    y_pos_label = y_max_plot-((y_max_plot-0)/100)*5
    x_pos_label = mean_value -((data[col].max() - data[col].min())/100)*2
    mean_df = pd.DataFrame({
                            'x_pos_label': [x_pos_label],
                            'y_pos_label': [y_pos_label],
                            'mean_label': [mean_label]
                           })

    plot = (
        p9.ggplot(data, p9.aes(x=col, y='..density..'))
        + p9.geom_histogram(
            bins=50,
            alpha=0.2,
            fill = color_general,
            color = color_general,
            position='identity'
        )
        +p9.geom_vline(
            xintercept = mean_value,
            color = color_general,
            linetype='dashed',
            size=0.8
        )
        +p9.geom_text(
            data = mean_df,
            mapping = p9.aes(
                x='x_pos_label',
                y='y_pos_label',
                label='mean_label'
            ),
            color=color_general,
            ha='right',
            size=5,
            show_legend = False
        )
        + p9.labs(
            title=f"Distribucion de '{col}'",
            x=col,
            y='Densidad'
        )        
        + p9.scale_y_continuous(expand=(0, 0), limits=(0,y_max_plot))
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

def get_grid_size (subplot_size,rows_grid,columns_grid):
    return (subplot_size[0]*columns_grid,subplot_size[1]*rows_grid)


def box_plot_continuous(data,continuous_columns,target_var,fig_size,colors):
    data_box_plot = pd.DataFrame({'y_value':[] ,'sector':[],'variable':[]})
    
    for col in continuous_columns:
        data_grouped = data[[col,target_var]].copy()
        data_grouped['variable']=col
        data_grouped = data_grouped.rename(columns={target_var: 'sector'})
        data_grouped = data_grouped.rename(columns={col: 'y_value'})
        data_grouped['sector'] = data_grouped['sector'].map({True:f'{target_var} True',False:f'{target_var} False'})
        
        data_general = data_grouped.copy()
        data_general['sector'] = 'General'
        
        data_general = pd.concat([data_general, data_grouped],ignore_index=True)
        data_box_plot = pd.concat([data_box_plot,data_general], ignore_index= True)
    
    data_box_plot['sector'] = pd.Categorical(
        data_box_plot['sector'],
        categories=['diabetes True','diabetes False','General'],
        ordered=True
    )  
    
    plot = (
        p9.ggplot(data_box_plot, p9.aes(
        x='sector',
        y='y_value',
        fill='sector',
        color='sector'
        )) +
        p9.geom_boxplot(outlier_alpha=0.1, 
                        alpha=0.2,
                        width=0.6,
                        position=p9.position_dodge(width=0.75))
        + p9.facet_wrap('~variable', ncol=1, scales='free')
        + p9.coord_flip()
        + p9.ggtitle('Continuous Variables boxplot')
        + p9.scale_fill_manual(values=colors)
        + p9.scale_color_manual(values=colors)
        + p9.theme(
            panel_background=p9.element_rect(fill="#ffffff"),
            plot_background=p9.element_rect(fill='#ffffff'),
            strip_background=p9.element_rect(fill='#ffffff'),
            axis_ticks_major_y = p9.element_blank(),
            axis_text_x=p9.element_text(size=7),
            axis_text_y=p9.element_blank(),
            axis_title_x=p9.element_blank(),
            axis_title_y=p9.element_blank(),
            plot_title=p9.element_text(size=12, weight="bold"),
            legend_title=p9.element_text(size=8),
            legend_text=p9.element_text(size=7),
            figure_size=fig_size
        )
    )

    return plot

