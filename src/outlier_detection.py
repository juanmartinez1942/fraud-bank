import pandas as pd

def remove_outliers_iqr(df, cols, target='isFraud'):
    """
    Elimina valores atípicos (outliers) de columnas numéricas usando el método IQR,
    pero solo para la clase mayoritaria (no fraudulenta) para preservar datos valiosos de fraude.

    Parameters:
        df (pd.DataFrame): Conjunto de datos original.
        cols (list): Lista de columnas numéricas sobre las que aplicar detección de outliers.
        target (str): Nombre de la variable objetivo binaria. Default: 'isFraud'.

    Returns:
        pd.DataFrame: Dataset combinado sin outliers en la clase legítima y con todos los casos de fraude.
    """
    df_fraud = df[df[target] == 1]
    df_legit = df[df[target] == 0]

    for col in cols:
        Q1 = df_legit[col].quantile(0.25)
        Q3 = df_legit[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_legit = df_legit[(df_legit[col] >= lower) & (df_legit[col] <= upper)]

    return pd.concat([df_legit, df_fraud], axis=0).sample(frac=1, random_state=42)