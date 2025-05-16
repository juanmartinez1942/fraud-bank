# src/model_section.py

import os
import pandas as pd
import plotly.express as px
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def load_monitoring_data(csv_path=os.getenv("MONITORING_METRICS")):
    """
    Carga el archivo CSV de métricas de monitoreo del modelo.

    Parameters:
        csv_path (str): Ruta al archivo CSV con métricas históricas.

    Returns:
        pd.DataFrame: DataFrame con columnas como 'date', 'auc', 'recall', etc.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si ocurre un error al leer el archivo.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"El archivo no existe: {csv_path}")
    try:
        return pd.read_csv(csv_path, parse_dates=["date"])
    except Exception as e:
        raise ValueError(f"Error al leer el archivo CSV: {e}")


def plot_model_performance(df):
    """
    Genera un gráfico de línea con la evolución de las métricas del modelo a lo largo del tiempo.

    Parameters:
        df (pd.DataFrame): DataFrame con columnas 'date', 'auc', 'precision', 'recall', 'f1_score'.

    Returns:
        plotly.graph_objects.Figure: Gráfico de líneas con las métricas.
    """
    fig = px.line(
        df, x="date", y=["auc", "precision", "recall", "f1_score"],
        markers=True, title="Evolución de Métricas del Modelo",
        labels={"value": "Métrica", "date": "Fecha"}
    )
    return fig


def plot_drift(df):
    """
    Genera un gráfico de línea para visualizar el drift diario y marca los días en los que se activó un reentrenamiento.

    Parameters:
        df (pd.DataFrame): DataFrame con columnas 'date', 'drift_score', y 'retrain_triggered'.

    Returns:
        plotly.graph_objects.Figure: Gráfico con línea de drift y puntos de reentreno.
    """
    fig = px.line(df, x="date", y="drift_score", title="Drift Score Diario", markers=True)
    retrain_df = df[df["retrain_triggered"] == 1]
    fig.add_scatter(
        x=retrain_df["date"],
        y=retrain_df["drift_score"],
        mode="markers", marker=dict(color="red", size=10),
        name="Retrain Trigger"
    )
    return fig


def load_shap_bar_plot(csv_path):
    """
    Carga un archivo CSV con valores de SHAP y genera un gráfico de barras horizontales
    con las variables más influyentes en la clase fraudulenta.

    Parameters:
        csv_path (str): Ruta al CSV con columnas 'feature' e 'importance'.

    Returns:
        plotly.graph_objects.Figure: Gráfico de barras de importancia de variables.
    """
    df = pd.read_csv(csv_path)
    fig = px.bar(
        df.sort_values(by='importance', ascending=True),
        x='importance', y='feature', orientation='h',
        title="Importancia Global de Variables (Clase Fraudulenta)",
        labels={"importance": "Importancia Media Absoluta", "feature": "Variable"},
        height=600
    )
    return fig


def load_latest_metrics(df):
    """
    Extrae las métricas más recientes desde el DataFrame de monitoreo.

    Parameters:
        df (pd.DataFrame): DataFrame con métricas por fecha.

    Returns:
        dict: Diccionario con las métricas del último día.

    Raises:
        ValueError: Si el DataFrame está vacío.
    """
    if df.empty:
        raise ValueError("El DataFrame de monitoreo está vacío. No se puede extraer la última métrica.")
    latest = df.iloc[-1]
    summary = {
        "date": latest["date"].date(),
        "auc": latest["auc"],
        "precision": latest["precision"],
        "recall": latest["recall"],
        "f1_score": latest["f1_score"],
        "drift_score": latest["drift_score"],
        "retrain": bool(latest["retrain_triggered"])
    }
    return summary