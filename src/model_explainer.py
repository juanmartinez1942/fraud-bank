# src/model_explainer.py

import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Evita que SHAP intente mostrar en entorno no interactivo
shap.initjs()

def load_model(model_path):
    """
    Carga un modelo previamente entrenado desde un archivo.

    Parameters:
        model_path (str): Ruta al archivo .pkl del modelo.

    Returns:
        sklearn.base.BaseEstimator: Modelo deserializado.
    """
    return joblib.load(model_path)


def get_shap_values(model, X_sample):
    """
    Calcula los valores SHAP para una muestra de datos utilizando TreeExplainer.

    Parameters:
        model (sklearn.base.BaseEstimator): Modelo ya entrenado.
        X_sample (pd.DataFrame): Subconjunto representativo del dataset para explicación.

    Returns:
        tuple: 
            - explainer (shap.TreeExplainer): Instancia del explicador.
            - shap_values (list or np.ndarray): Valores SHAP generados.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def plot_shap_summary_bar(shap_values, X_sample, class_idx=1):
    """
    Genera un gráfico de barras que muestra la importancia global media de cada feature.

    Parameters:
        shap_values (list or np.ndarray): Matriz de valores SHAP por clase.
        X_sample (pd.DataFrame): Datos correspondientes a los SHAP values.
        class_idx (int): Índice de la clase objetivo (por defecto, 1 = fraude).

    Returns:
        matplotlib.figure.Figure: Figura matplotlib lista para guardar o mostrar.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[class_idx], X_sample, plot_type="bar", show=False)
    return plt.gcf()  # Devuelve la figura actual


def plot_shap_summary_dot(shap_values, X_sample, class_idx=1):
    """
    Genera un gráfico de dispersión SHAP (dot plot) que muestra la distribución del impacto 
    de cada feature sobre la predicción.

    Parameters:
        shap_values (list or np.ndarray): Matriz de valores SHAP por clase.
        X_sample (pd.DataFrame): Datos correspondientes a los SHAP values.
        class_idx (int): Índice de la clase objetivo (por defecto, 1 = fraude).

    Returns:
        matplotlib.figure.Figure: Figura matplotlib con el gráfico generado.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[class_idx], X_sample, plot_type="dot", show=False)
    return plt.gcf()