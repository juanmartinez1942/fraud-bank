import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(X, y, save_path=None):
    """
    Entrena un modelo Random Forest sobre los datos proporcionados y opcionalmente guarda el modelo.

    Parameters:
        X (pd.DataFrame): Variables predictoras.
        y (pd.Series): Variable objetivo binaria.
        save_path (str, optional): Ruta donde guardar el modelo entrenado (.pkl). Si no se proporciona, no se guarda.

    Returns:
        tuple: 
            - model (RandomForestClassifier): Modelo entrenado.
            - X_train (pd.DataFrame): Subconjunto de entrenamiento.
            - X_test (pd.DataFrame): Subconjunto de prueba.
            - y_train (pd.Series): Target de entrenamiento.
            - y_test (pd.Series): Target de prueba.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    if save_path:
        joblib.dump(model, save_path)

    return model, X_train, X_test, y_train, y_test