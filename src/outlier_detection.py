import pandas as pd

def remove_outliers_iqr(df, cols, target='isFraud'):
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