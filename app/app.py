# app.py

import streamlit as st
import pandas as pd
from src import eda_section, model_section
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuración ---
st.set_page_config(layout="wide")

# --- Cargar datos dinámicamente ---
st.sidebar.header("📁 Cargar Dataset")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file).sample(n=200_000, random_state=42)
    st.sidebar.success("✅ Dataset cargado correctamente.")
else:
    st.sidebar.info("📌 Usando dataset por defecto.")
    df = pd.read_csv(os.getenv("DEFAULT_DATA_PATH")).sample(n=int(os.getenv("SAMPLE_SIZE")), random_state=int(os.getenv("RANDOM_STATE")))

# --- Tabs ---
tabs = st.tabs(["📊 EDA", "🧠 Modelo y Monitoreo"])

# --- Tab 1: EDA ---
with tabs[0]:
    st.title("📊 Exploración de Transacciones Bancarias")
    
    # Información general
    info = eda_section.basic_info(df)
    st.write("Dimensiones:", info["shape"])
    st.write("Tipos de datos:")
    st.dataframe(info["dtypes"])
    st.write("Valores nulos:")
    st.dataframe(info["missing"])

    # Distribución de fraude
    st.sidebar.markdown("### 🎯 Variable objetivo (target)")
    default_target = "isFraud" if "isFraud" in df.columns else df.columns[-1]
    target_var = st.sidebar.selectbox("Selecciona la variable objetivo", df.columns.tolist(), index=df.columns.get_loc(default_target))

    # Mostrar distribución si el target es binario
    if df[target_var].nunique() <= 2:
        dist = eda_section.fraud_distribution(df, target_var)
        if dist is not None:
            st.plotly_chart(eda_section.plot_fraud_pie(dist))
    else:
        st.info(f"La variable '{target_var}' no parece binaria, por lo que no se mostrará el gráfico de distribución.")

    st.markdown("---")
    st.subheader("📊 Gráfico de barras por variable categórica")
    cat_vars = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_vars:
        cat_var = st.selectbox("Variable categórica", cat_vars)
        color_by = st.selectbox("¿Colorear por variable?", [None, target_var] + cat_vars)
        st.plotly_chart(eda_section.plot_categorical_bar(df, cat_var, color_by))
    else:
        st.warning("No se encontraron variables categóricas.")

    st.markdown("---")
    st.subheader("📈 Distribución de variable numérica")
    num_vars = df.select_dtypes(include="number").columns.tolist()
    if num_vars:
        num_var = st.selectbox("Variable numérica", num_vars, key="dist")
        hue = st.selectbox("¿Separar por categoría?", [None, target_var] + cat_vars, key="hue_dist")
        st.plotly_chart(eda_section.plot_numeric_distribution(df, num_var, hue))
    else:
        st.warning("No se encontraron variables numéricas.")

    st.markdown("---")
    st.subheader("📦 Boxplot por categoría")
    box_y = st.selectbox("Variable numérica", num_vars, key="box_y")
    box_x_candidates = [v for v in cat_vars if df[v].nunique() < 20]
    if box_x_candidates:
        box_x = st.selectbox("Categoría (con pocos valores)", box_x_candidates, key="box_x")
        st.plotly_chart(eda_section.plot_box_by_category(df, box_y, box_x))
    else:
        st.warning("No hay variables categóricas con pocos valores.")

    st.markdown("---")
    st.subheader("🔗 Matriz de Correlación Numérica")
    st.plotly_chart(eda_section.plot_corr_heatmap(df))

# --- Tab 2: Modelo y Monitoreo ---
with tabs[1]:
    st.title("🧠 Monitoreo del Modelo")

    # Mostrar warning si hay drift o baja performance
    warning_path = os.getenv("WARNING_FILE")
    if os.path.exists(warning_path):
        with open(warning_path, "r") as f:
            warning = f.read().strip()
        if warning:
            st.warning(warning)

    monitor_df = model_section.load_monitoring_data(os.getenv("MONITORING_METRICS"))
    st.plotly_chart(model_section.plot_model_performance(monitor_df))
    st.plotly_chart(model_section.plot_drift(monitor_df))

    st.markdown("""
    **ℹ️ ¿Qué significa el Drift Score?**

    El *drift score* mide cuánto se han desviado las características actuales respecto a los datos con los que se entrenó el modelo. 
    Un valor más alto indica un cambio significativo en la distribución de los datos, lo que podría deteriorar el rendimiento del modelo.

    - Drift bajo (≤ 0.10): distribución estable  
    - Drift moderado (0.10–0.15): monitorear  
    - Drift alto (> 0.15): potencial necesidad de reentrenar el modelo
    """)

    st.subheader("📌 Últimas Métricas del Modelo")
    latest = model_section.load_latest_metrics(monitor_df)
    st.markdown(f"""
    - Fecha: `{latest['date']}`  
    - AUC: `{latest['auc']:.3f}`  
    - Precision: `{latest['precision']:.3f}`  
    - Recall: `{latest['recall']:.3f}`  
    - F1 Score: `{latest['f1_score']:.3f}`  
    - Drift Score: `{latest['drift_score']:.3f}`  
    - Reentrenamiento: `{'✅ Sí' if latest['retrain'] else '❌ No'}`
    """)

    st.subheader("📊 SHAP - Importancia Global de Variables")
    st.plotly_chart(model_section.load_shap_bar_plot(os.getenv("SHAP_GLOBAL")))