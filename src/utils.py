import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from src.data_preprocessing import engineer_features, impute_missing_values
from src.model_training import train_model
from src.model_section import load_monitoring_data
from src.monitoring import check_drift
from dotenv import load_dotenv

load_dotenv()

# 🚨 Condición para detectar drift o baja performance
def check_drift_condition(metrics, auc_threshold=0.88, drift_threshold=0.15):
    return metrics["auc"] < auc_threshold or metrics["drift_score"] > drift_threshold

# ⚠️ Guarda un warning visible desde Streamlit
def write_retrain_warning(flag, warning_path="data/retrain_warning.txt"):
    with open(warning_path, "w") as f:
        if flag:
            f.write("⚠️ Se recomienda reentrenar el modelo: drift o bajo rendimiento detectado.")
        else:
            f.write("")

# 🗓 Simula cron semanal - entrena el modelo con última data
def cron_weekly_train_model():
    df = pd.read_csv(os.getenv("FRAUD_DATASET"))
    df = engineer_features(df)
    df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])
    num_cols = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest',
        'balance_diff_orig', 'balance_diff_dest',
        'amount_to_balance_ratio'
    ]
    df = impute_missing_values(df, num_cols)
    model, *_ = train_model(df.drop(columns=["isFraud"]), df["isFraud"], save_path=os.getenv("MODEL_PATH"))


    # Check drift + performance y escribir warning
    latest = load_monitoring_data(os.getenv("MONITORING_METRICS")).iloc[-1]
    flag = check_drift_condition(latest)
    write_retrain_warning(flag)

# 🗓 Simula cron diario - genera predicciones
def cron_daily_predict():
    model = joblib.load(os.getenv("MODEL_PATH"))
    df = pd.read_csv(os.getenv("FRAUD_DATASET"))
    df = engineer_features(df)
    df = impute_missing_values(df, [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest',
        'balance_diff_orig', 'balance_diff_dest', 'amount_to_balance_ratio'
    ])
    X = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud", "isFraud"])
    df["pred_proba"] = model.predict_proba(X)[:, 1]
    df["pred_label"] = model.predict(X)

    today = datetime.today().strftime("%Y-%m-%d")
    df.to_csv(f"data/predictions_{today}.csv", index=False)

# 🗓 Simula cron diario - evalúa y actualiza monitoring_metrics.csv
def cron_daily_evaluate():
    today = datetime.today().strftime("%Y-%m-%d")
    df = pd.read_csv(f"data/predictions_{today}.csv")
    df_ref = pd.read_csv(os.getenv("FRAUD_DATASET"))

    # Seleccionar columnas para evaluar drift
    drift_cols = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest',
        'balance_diff_orig', 'balance_diff_dest',
        'amount_to_balance_ratio'
    ]

    drift_report = check_drift(df[drift_cols], df_ref[drift_cols], columns=drift_cols)
    
    # Calcular drift_score promedio
    avg_drift_score = sum(d["drift_score"] for d in drift_report.values()) / len(drift_report)

    metrics = {
        "date": today,
        "auc": roc_auc_score(df["isFraud"], df["pred_proba"]),
        "precision": precision_score(df["isFraud"], df["pred_label"]),
        "recall": recall_score(df["isFraud"], df["pred_label"]),
        "f1_score": f1_score(df["isFraud"], df["pred_label"]),
        "drift_score": avg_drift_score,
        "retrain_triggered": int(check_drift_condition({
            "auc": roc_auc_score(df["isFraud"], df["pred_proba"]),
            "drift_score": avg_drift_score
        }))
    }

    monitor_df = pd.read_csv(os.getenv("MONITORING_METRICS"))
    monitor_df = pd.concat([monitor_df, pd.DataFrame([metrics])], ignore_index=True)
    monitor_df.to_csv(os.getenv("MONITORING_METRICS"), index=False)