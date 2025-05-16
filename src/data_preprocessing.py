# src/data_preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer
import joblib

def engineer_features(df):
    """
    Realiza ingeniería de características sobre el dataset original.

    Crea nuevas variables relevantes para la detección de fraude:
        - Diferencias de saldo antes y después de la transacción.
        - Indicadores de errores o inconsistencias en saldos (como saldos 0 antes y después).
        - Ratio entre monto y saldo original.
        - Codificación one-hot del tipo de transacción.

    Parameters:
        df (pd.DataFrame): Dataset original con columnas financieras y de tipo de transacción.

    Returns:
        pd.DataFrame: Dataset enriquecido con nuevas columnas.
    """
    df = df.copy()

    # 1. Diferencias de saldo
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

    # 2. Errores o casos atípicos
    # Marcamos transacciones donde el saldo inicial y final son 0
    df['error_balance_orig'] = ((df['oldbalanceOrg'] == 0) & (df['newbalanceOrig'] == 0)).astype(int)
    df['error_balance_dest'] = ((df['oldbalanceDest'] == 0) & (df['newbalanceDest'] == 0)).astype(int)

    # 3. Ratio monto / saldo
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)

    # 4. One-hot encoding del tipo de transacción
    df = pd.get_dummies(df, columns=['type'], prefix='type')

    return df


def impute_missing_values(X, num_cols, strategy='median', save_path=None):
    """
    Imputa valores faltantes en columnas numéricas utilizando una estrategia definida.

    Parameters:
        X (pd.DataFrame): Dataset a imputar.
        num_cols (list): Lista de nombres de columnas numéricas a procesar.
        strategy (str): Estrategia de imputación ('mean', 'median', etc.).
        save_path (str, optional): Ruta para guardar el imputador como archivo .pkl.

    Returns:
        pd.DataFrame: Dataset con valores imputados.
    """
    imputer = SimpleImputer(strategy=strategy)
    X[num_cols] = imputer.fit_transform(X[num_cols])
    if save_path:
        joblib.dump(imputer, save_path)
    return X