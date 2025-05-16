from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
import os

load_dotenv()
def check_drift(df_new, df_ref, columns, threshold=0.1):
    drift_report = {}
    for col in columns:
        dist = wasserstein_distance(df_new[col], df_ref[col])
        drift_report[col] = {
            "drift_score": dist,
            "drifted": dist > threshold
        }
    return drift_report

def check_model_degradation(y_true, y_proba, baseline_auc, threshold=0.05):
    current_auc = roc_auc_score(y_true, y_proba)
    auc_drop = baseline_auc - current_auc
    needs_retraining = auc_drop > threshold
    return {
        "current_auc": current_auc,
        "auc_drop": auc_drop,
        "retrain": needs_retraining
    }