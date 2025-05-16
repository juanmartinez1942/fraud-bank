# src/model_explainer.py

import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Evita que SHAP intente mostrar en entorno no interactivo
shap.initjs()


def load_model(model_path):
    return joblib.load(model_path)


def get_shap_values(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def plot_shap_summary_bar(shap_values, X_sample, class_idx=1):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[class_idx], X_sample, plot_type="bar", show=False)
    return plt.gcf()  # Devuelve figura actual para usar en Streamlit


def plot_shap_summary_dot(shap_values, X_sample, class_idx=1):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[class_idx], X_sample, plot_type="dot", show=False)
    return plt.gcf()