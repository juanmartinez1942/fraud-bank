import plotly.express as px
import plotly.graph_objects as go

def basic_info(df):
    """
    Devuelve información básica del DataFrame, incluyendo su forma, tipos de datos y cantidad de valores nulos por columna.

    Args:
        df (pd.DataFrame): DataFrame a analizar.

    Returns:
        dict: Diccionario con las claves 'shape', 'dtypes' y 'missing'.
    """
    info = {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "missing": df.isnull().sum(),
    }
    return info

def fraud_distribution(df, target_var):
    """
    Calcula la distribución porcentual de la variable objetivo (fraude/no fraude).

    Args:
        df (pd.DataFrame): DataFrame a analizar.
        target_var (str): Nombre de la columna objetivo.

    Returns:
        pd.Series or None: Serie con el porcentaje de cada clase o None si la columna no existe.
    """
    if target_var in df.columns:
        return df[target_var].value_counts(normalize=True) * 100
    return None

def plot_fraud_pie(dist):
    """
    Genera un gráfico de torta (pie) con la distribución de fraude y no fraude.

    Args:
        dist (pd.Series): Serie con la distribución porcentual de fraude.

    Returns:
        plotly.graph_objs._figure.Figure: Figura de Plotly.
    """
    return px.pie(
        values=dist.values,
        names=dist.index.map({0: "No Fraud", 1: "Fraud"}),
        title="Porcentaje de Fraude"
    )

def plot_categorical_bar(df, cat_var, color_by=None):
    """
    Genera un histograma de barras para una variable categórica, opcionalmente coloreado por otra variable.

    Args:
        df (pd.DataFrame): DataFrame a analizar.
        cat_var (str): Nombre de la variable categórica.
        color_by (str, opcional): Variable para colorear las barras.

    Returns:
        plotly.graph_objs._figure.Figure: Figura de Plotly.
    """
    fig = px.histogram(df, x=cat_var, color=color_by, barmode="group")
    fig.update_layout(title=f"Distribución de {cat_var}")
    return fig

def plot_numeric_distribution(df, num_var, hue=None):
    """
    Genera un histograma y boxplot para una variable numérica, opcionalmente segmentado por una variable categórica.

    Args:
        df (pd.DataFrame): DataFrame a analizar.
        num_var (str): Nombre de la variable numérica.
        hue (str, opcional): Variable para segmentar el color.

    Returns:
        plotly.graph_objs._figure.Figure: Figura de Plotly.
    """
    fig = px.histogram(df, x=num_var, color=hue, marginal="box", barmode="overlay")
    fig.update_layout(title=f"Distribución de {num_var}")
    return fig

def plot_box_by_category(df, num_var, cat_var):
    """
    Genera un boxplot de una variable numérica segmentada por una variable categórica.

    Args:
        df (pd.DataFrame): DataFrame a analizar.
        num_var (str): Variable numérica.
        cat_var (str): Variable categórica.

    Returns:
        plotly.graph_objs._figure.Figure: Figura de Plotly.
    """
    fig = px.box(df, x=cat_var, y=num_var, points="outliers", color=cat_var)
    fig.update_layout(title=f"{num_var} por {cat_var}")
    return fig

def plot_corr_heatmap(df):
    """
    Genera un mapa de calor (heatmap) de la matriz de correlación entre variables numéricas.

    Args:
        df (pd.DataFrame): DataFrame a analizar.

    Returns:
        plotly.graph_objs._figure.Figure: Figura de Plotly.
    """
    num_vars = df.select_dtypes(include="number").columns.tolist()
    corr = df[num_vars].corr()
    fig = px.imshow(corr, text_auto=True, title="Matriz de Correlación Numérica")
    return fig