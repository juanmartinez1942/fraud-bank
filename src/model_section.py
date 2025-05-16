# src/model_section.py

import os
import pandas as pd
import plotly.express as px
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def load_monitoring_data(csv_path=os.getenv("MONITORING_METRICS")):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"El archivo no existe: {csv_path}")
    try:
        return pd.read_csv(csv_path, parse_dates=["date"])
    except Exception as e:
        raise ValueError(f"Error al leer el archivo CSV: {e}")

def plot_model_performance(df):
    fig = px.line(
        df, x="date", y=["auc", "precision", "recall", "f1_score"],
        markers=True, title="Evolución de Métricas del Modelo",
        labels={"value": "Métrica", "date": "Fecha"}
    )
    return fig

def plot_drift(df):
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