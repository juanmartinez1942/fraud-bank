import plotly.express as px
import plotly.graph_objects as go

def basic_info(df):
    info = {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "missing": df.isnull().sum(),
    }
    return info

def fraud_distribution(df, target_var):
    if target_var in df.columns:
        return df[target_var].value_counts(normalize=True) * 100
    return None

def plot_fraud_pie(dist):
    return px.pie(
        values=dist.values,
        names=dist.index.map({0: "No Fraud", 1: "Fraud"}),
        title="Porcentaje de Fraude"
    )

def plot_categorical_bar(df, cat_var, color_by=None):
    fig = px.histogram(df, x=cat_var, color=color_by, barmode="group")
    fig.update_layout(title=f"Distribución de {cat_var}")
    return fig

def plot_numeric_distribution(df, num_var, hue=None):
    fig = px.histogram(df, x=num_var, color=hue, marginal="box", barmode="overlay")
    fig.update_layout(title=f"Distribución de {num_var}")
    return fig

def plot_box_by_category(df, num_var, cat_var):
    fig = px.box(df, x=cat_var, y=num_var, points="outliers", color=cat_var)
    fig.update_layout(title=f"{num_var} por {cat_var}")
    return fig

def plot_corr_heatmap(df):
    num_vars = df.select_dtypes(include="number").columns.tolist()
    corr = df[num_vars].corr()
    fig = px.imshow(corr, text_auto=True, title="Matriz de Correlación Numérica")
    return fig