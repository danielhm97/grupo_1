import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from scipy.stats import norm
from scipy import stats
from optbinning import OptimalBinning
from statsmodels.stats.outliers_influence import variance_inflation_factor
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter
import os
from sklearn.metrics import roc_curve, accuracy_score, f1_score

def hessian(x, p):
    """
    Function that calculates Hessian matrix (h)
    p: scalar with conditional probability of default
    w: wight based on probability p
    """
    w = p * (1 - p)
    h = np.dot(x.T * w, x)
    return h

def model_summary(X_train, model_coef):
    """
    Esta función devuelve el resumen de un modelo de regresión incluyendo los coeficientes,
    los p-valores y el VIF de cada variable.

    Parameters
    ----------
    X_train : DataFrame
        Tabla usada para ajustar el modelo.
    model_coef : DataFrame
        DataFrame con una columna llamada 'Coeficientes' que contenga los coeficientes de cada variable.

    Returns
    -------
    model_coef : DataFrame
        Tabla que contiene el resumen del modelo con campos como Variable, Coeficiente, p-value, VIF...

    """
    coef = model_coef['Coeficiente'].tolist()
    x_intercept = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    linear_pred = x_intercept.dot(coef)
    prob_pred = 1 / (1 + np.exp(-linear_pred))
    h = hessian(x_intercept, prob_pred)
    h_inv = np.linalg.inv(h)
    # Standard error
    standard_errors = np.sqrt(np.diag(h_inv))
    model_coef['std_err'] = standard_errors
    # z-value
    z_values = coef / standard_errors
    model_coef['z_value'] = z_values
    # p-value
    p_values = 2 * (1 - norm.cdf(np.abs(z_values)))
    model_coef['p_value'] = p_values
    # VIF
    model_coef["VIF"] = [variance_inflation_factor(x_intercept, i) for i in range(x_intercept.shape[1])]
    return model_coef

def backward_selector(X_train, y_train, X_test, y_test, model, feature_fix, cv_folds=5, seed=42):
    """
    Función que ejecuta el algoritmo de backward selector. Este algoritmo consiste en ir eliminando variables
    una a una hasta que el modelo se entrene únicamente con una variable. La variable eliminada en cada
    iteración es aquella cuya eliminación hace que el AUC del modelo baje lo mínimo.

    Parameters
    ----------
    X_train : DataFrame
        Tabla usada para ajustar el modelo.
    y_train : Series
        Variable target correspondiente al conjunto de datos de entrenamiento.
    X_test : DataFrame
        Tabla usada para validar el modelo.
    y_test : Series
        Variable target correspondiente al conjunto de datos de validación.
    model : modelo
        Modelo a usar en el algoritmo de backward selector (regresión).
    feature_fix : str
        Nombre de una posible variable que fijar mantener en todos los modelos.
    cv_folds : int, optional
        Número de folds con los que validar los modelos. The default is 5.
    seed : int, optional
        Semilla para poder reproducir los resultados. The default is 42.

    Returns
    -------
    results_val : Diccionario
        Diccionario con los resultados de validación.
    results_test : Diccionario
        Diccionario con los resultados de test.

    """
    features = list(X_train.columns)
    best_features = features
    results_val = {}
    results_test = {}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    while len(best_features) > 0:
        aucs_val = []

        # Evaluamos con cross-validation
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_train_cv, X_val_cv = X_train.iloc[train_idx][best_features], X_train.iloc[val_idx][best_features]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            clf = clone(model)
            clf.fit(X_train_cv, y_train_cv)
            y_pred_cv = clf.predict_proba(X_val_cv)[:, 1]
            aucs_val.append(roc_auc_score(y_val_cv, y_pred_cv))

        results_val[len(best_features)] = {'features': best_features,
                                           'aucs': aucs_val}

        # Ahora entrenamos con todos los datos y evaluamos en test
        clf = clone(model)
        clf.fit(X_train[best_features], y_train)

        coeficientes = pd.DataFrame({'Variable' : best_features, 'Coeficiente' : clf.coef_[0]})
        coeficientes.loc[coeficientes.shape[0]] = ['intercepto', clf.intercept_[0]]
        coeficientes = model_summary(X_train[best_features], coeficientes)

        y_pred_test = clf.predict_proba(X_test[best_features])[:, 1]
        auc_test = roc_auc_score(y_test, y_pred_test)

        results_test[len(best_features)] = {'features': best_features,
                                           'model': clf,
                                           'model_summary': coeficientes,
                                           'auc': auc_test}

        if len(best_features) == 1:
            break

        # Buscar la peor variable para eliminar
        worst_feature = None
        best_auc = float('-inf')

        for feature in [feature for feature in best_features if feature != feature_fix]:
            temp_features = best_features.copy()
            temp_features.remove(feature)

            aucs_temp = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_train_cv, X_val_cv = X_train.iloc[train_idx][temp_features], X_train.iloc[val_idx][temp_features]
                y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

                clf = clone(model)
                clf.fit(X_train_cv, y_train_cv)
                y_pred = clf.predict_proba(X_val_cv)[:, 1]
                aucs_temp.append(roc_auc_score(y_val_cv, y_pred))

            avg_auc = np.mean(aucs_temp)

            if avg_auc > best_auc:
                best_auc = avg_auc
                worst_feature = feature

        best_features = [feature for feature in best_features if feature != worst_feature]  # Eliminamos la peor variable

    return results_val, results_test

def plot_results_val(results_val):
    """
    Función que pinta un gráfico con los resultados del modelo en validación.
    Se ve como varía la capacidad predictiva y la variabilidad de la misma en
    función del número de variables incluidas en el modelo.

    Parameters
    ----------
    results_val : Diccionario
        Output de la función backward_selector.

    Returns
    -------
    None.

    """
    df_val = pd.DataFrame(dict([(k, pd.Series(results_val[k]['aucs'])) for k in sorted(results_val.keys())]))

    plt.figure(figsize=(8, 5))

    bp = df_val.boxplot(
                    grid=True, patch_artist=True, 
                    boxprops=dict(color="black", facecolor="white"),
                    medianprops=dict(color="black", linewidth=1),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    flierprops=dict(marker="o", color="black", markersize=5)
                    )

    plt.xlabel("Número de variables")
    plt.ylabel("AUC")
    plt.title("Variabilidad de AUC de validación por número de variables")
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    plt.show()

def plot_results_test(results_test):
    """
    Función que pinta un gráfico con los resultados del modelo en test.
    Se ve como varía la capacidad predictiva en test en
    función del número de variables incluidas en el modelo.

    Parameters
    ----------
    results_test : Diccionario
        Output de la función backward_selector.

    Returns
    -------
    None.

    """
    num_features = sorted(results_test.keys())
    aucs = [results_test[n_features]['auc'] for n_features in num_features]
    num_features = [str(n_features) for n_features in num_features]

    plt.figure(figsize=(8, 5))

    plt.plot(num_features, aucs, marker='o', linestyle='-', color='black')

    plt.xlabel("Número de variables")
    plt.ylabel("AUC")
    plt.title("Variabilidad de AUC de test por número de variables")
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    plt.show()

def sorted_correlation_matrix_IV(data_woe, data_iv, threshold, method='spearman'):
    """
    Función que, si dos variables están correladas, elimina aquella cuya
    capacidad predictiva es menor (en función del IV)

    Parameters
    ----------
    data_woe : DataFrame
        Tabla con las variables trameadas y codificadas mediante el WOE
    data_iv : DataFrame
        Tabla con el IV de cada variable
    threshold : float
        Punto de corte a partir del cual considerar dos variables correladas
    method : str, optional
        Tipo de correlación que usar. The default is 'spearman'.

    Returns
    -------
    df_sorted : DataFrame
        Tabla sin variables correladas.

    """
    data_corr = data_woe.corr(method=method)
    data_corr = data_corr.reset_index()
    data_corr = data_corr.rename(columns={'index': 'Variable'})
    df_corr_iv = data_corr.merge(data_iv, on='Variable',how='left')
    df_corr_iv = df_corr_iv.set_index('Variable')
           
    # Sort the correlation matrix by IV value in descending order
    df_sort_IV = df_corr_iv.sort_values(by='IV', ascending=False)
    
    # Rearrange the columns of the correlation matrix so that there are only 1's in the diagonal.
    df_sorted = df_sort_IV[df_sort_IV.index.tolist()]

    mask = df_sorted.where(np.triu(np.ones(df_sorted.shape), k=1).astype(bool))
    for col in mask.columns:
        max_corr = mask[col].max()
        if threshold < abs(max_corr):
            mask = mask.drop(col, axis=1).drop(col, axis=0)

    df_sorted = data_woe[mask.columns]
    
    return df_sorted

def discretize_variables(data, data_test, variables, categorical_vars, continuous_vars, target):
    """
    Función que discretiza las variables basándose en un
    algoritmo basado en árboles que busca los puntos de 
    corte óptimos. Calcula los puntos de corte con data
    y los aplica a data y data_test.

    Parameters
    ----------
    data : DataFrame
        Datos de train.
    data_test : DataFrame
        Datos de test.
    variables : list
        Lista de variables a discretizar.
    target : str
        Nombre variable target.

    Returns
    -------
    data_disc : DataFrame
        Datos de train discretizados.
    data_test_disc : DataFrame
        Datos de test discretizados.
    data_iv : DataFrame
        Tabla con los IVs de cada variable.

    """
    data_disc = data.copy()
    data_test_disc = data_test.copy()
    data_iv = pd.DataFrame({'Variable':[], 'IV':[]})
    for var in variables:
        if var in data.columns:
            if var in categorical_vars:
                data_disc, data_test_disc, iv = var_discretizer(data_disc, data_test_disc, var, 'categorical', target)
                data_iv.loc[data_iv.shape[0]] = [var, iv]
            if var in continuous_vars:
                data_disc, data_test_disc, iv = var_discretizer(data_disc, data_test_disc, var, 'numerical', target)
                data_iv.loc[data_iv.shape[0]] = [var, iv]
    data_iv = data_iv.sort_values(by='IV', ascending=False)
    return data_disc, data_test_disc, data_iv

def var_discretizer(data, data_test, var, var_type, target):
    """
    Función que discretiza una variable.

    Parameters
    ----------
    data : DataFrame
        Datos de train.
    data_test : DataFrame
        Datos de test.
    var : str
        Variable a discretizar.
    var_type : str
        Tipo de la variable (categorical o numerical).
    target : str
        Nombre variable target.

    Returns
    -------
    data : DataFrame
        Datos de train con la variable discretizada.
    data_test : DataFrame
        Datos de test con la variable discretizada.
    iv : float
        Poder predictivo de la variable

    """
    optb = OptimalBinning(name=var, dtype=var_type, min_bin_size=0.05)
    optb.fit(data[var].values, data[target])
    table = optb.binning_table.build()
    iv = table.loc['Totals', 'IV']
    data[var] = optb.transform(data[var].values, metric='bins')
    data_test[var] = optb.transform(data_test[var].values, metric='bins')
    return data, data_test, iv

def plot_business_sense(data, feature, target, category=True, q=0):
    """
    Función que pinta un gráfico para analizar el sentido de
    negocio de cada variable.

    Parameters
    ----------
    data : DataFrame
        Tabla de datos.
    feature : str
        Nombre de variable a analizar.
    target : str
        Nombre de la target.
    category : float, optional
        Indica si la variable es categórica o no. The default is True.
    q : int, optional
        Numero de bins que graficar. The default is 0.

    Returns
    -------
    None.

    """
    if category == False and q == 0:
        plot_data = data[[feature, target]].copy()
        bin_width = 0.05
        min_val = data[feature].min()
        max_val = data[feature].max()
        bins = np.arange(min_val, max_val + bin_width, bin_width)
        plot_data[feature] = pd.cut(plot_data[feature], bins=bins, right=False)
        count_data = (plot_data[feature].value_counts(normalize=True) * 100).reset_index()
        plot_data = (plot_data.groupby(feature)[target].mean() * 100).reset_index()
        plot_data.dropna(inplace=True)
        plot_data = plot_data.merge(count_data, how='inner', on=feature)
        plot_data[feature] = plot_data[feature].astype(str)
    if category == False and q > 0:
        plot_data = data[[feature, target]].copy()
        plot_data[feature] = pd.qcut(plot_data[feature], q=q, duplicates='drop')
        count_data = (plot_data[feature].value_counts(normalize=True) * 100).reset_index()
        plot_data = (plot_data.groupby(feature)[target].mean() * 100).reset_index()
        plot_data.dropna(inplace=True)
        plot_data = plot_data.merge(count_data, how='inner', on=feature)
        plot_data[feature] = plot_data[feature].astype(str)
    if category == True:
        plot_data = (data.groupby(feature)[target].mean() * 100).reset_index()
        count_data = (data[feature].value_counts(normalize=True) * 100).reset_index()
        plot_data = plot_data.merge(count_data, how='inner', on=feature)
    # Sort categories
    try:
        plot_data = plot_data.sort_values(by=feature,
                                        key=lambda x: x.apply(lambda x: float('inf') if x=='Missing' else float(x.split(',')[0][1:]))
                                        ).reset_index(drop=True)
    except:
        plot_data = plot_data.sort_values(by=feature,
                                        key=lambda x: x.map(lambda x: 'ZZZZZZ' if x == 'Missing' else x.strip("[").strip("]").strip("'"))
                                        ).reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.bar(plot_data[feature], plot_data['proportion'], color='#0085CA')
    ax1.set_ylabel('Porcentaje de registros (%)', color='#0085CA')
    ax1.set_xlabel(f'Categorías variable {feature}')
    ax1.set_title(f'Sentido en el modelo de la variable {feature}')
    ax1.tick_params(axis='y', labelcolor='#0085CA')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.plot(plot_data[feature], plot_data[target], color='black', marker='o', linestyle='-')
    ax2.set_ylabel('Porcentaje de target=1 (%)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    ax1.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()