# src/data_preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer
import joblib

def engineer_features(df):
    df = df.copy()

    # 1. Diferencias de saldo
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

    # 2. Errores o casos atípicos
        # Marcamos transacciones donde:
        # - El cliente tenía saldo 0 y quedó con saldo 0 → puede indicar transacción inválida o simulada.
        # - El destinatario tenía saldo 0 y siguió con saldo 0 → posible transacción fantasma.
    df['error_balance_orig'] = ((df['oldbalanceOrg'] == 0) & (df['newbalanceOrig'] == 0)).astype(int)
    df['error_balance_dest'] = ((df['oldbalanceDest'] == 0) & (df['newbalanceDest'] == 0)).astype(int)

    # 3. Ratio monto / saldo
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)

    # 4. One-hot encoding de tipo
    df = pd.get_dummies(df, columns=['type'], prefix='type')

    return df

def impute_missing_values(X, num_cols, strategy='median', save_path=None):
    imputer = SimpleImputer(strategy=strategy)
    X[num_cols] = imputer.fit_transform(X[num_cols])
    if save_path:
        joblib.dump(imputer, save_path)
    return X