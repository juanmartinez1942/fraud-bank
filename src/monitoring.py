from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
import os

load_dotenv()
def check_drift(df_new, df_ref, columns, threshold=0.1):
    """
    Calcula el drift (desviación) entre las distribuciones de las columnas seleccionadas
    de dos dataframes usando la distancia de Wasserstein.

    Args:
        df_new (pd.DataFrame): DataFrame con los datos actuales o nuevos.
        df_ref (pd.DataFrame): DataFrame con los datos de referencia (históricos).
        columns (list): Lista de nombres de columnas numéricas a comparar.
        threshold (float, opcional): Umbral a partir del cual se considera que hay drift. Por defecto 0.1.

    Returns:
        dict: Diccionario donde cada clave es una columna y el valor es otro diccionario con:
            - "drift_score": valor de la distancia de Wasserstein.
            - "drifted": True si el drift supera el umbral, False en caso contrario.
    """
    drift_report = {}
    for col in columns:
        dist = wasserstein_distance(df_new[col], df_ref[col])
        drift_report[col] = {
            "drift_score": dist,
            "drifted": dist > threshold
        }
    return drift_report

def check_model_degradation(y_true, y_proba, baseline_auc, threshold=0.05):
    """
    Evalúa la degradación del modelo comparando el AUC actual con el AUC de referencia (baseline).

    Args:
        y_true (array-like): Valores verdaderos de la variable objetivo.
        y_proba (array-like): Probabilidades predichas por el modelo.
        baseline_auc (float): Valor de AUC de referencia (baseline).
        threshold (float, opcional): Umbral de caída de AUC a partir del cual se recomienda reentrenar. Por defecto 0.05.

    Returns:
        dict: Diccionario con las métricas de evaluación:
            - "current_auc": AUC actual calculado.
            - "auc_drop": Diferencia entre el baseline y el AUC actual.
            - "retrain": True si la caída de AUC supera el umbral, False en caso contrario.
    """
    current_auc = roc_auc_score(y_true, y_proba)
    auc_drop = baseline_auc - current_auc
    needs_retraining = auc_drop > threshold
    return {
        "current_auc": current_auc,
        "auc_drop": auc_drop,
        "retrain": needs_retraining
    }