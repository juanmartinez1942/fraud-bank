# 🛡️ Fraud Detection - Prueba Técnica

Este proyecto tiene como objetivo detectar transacciones bancarias fraudulentas a partir de un conjunto de datos simulado. Se ha diseñado e implementado un sistema completo de análisis, entrenamiento de modelos, evaluación y monitoreo, con una interfaz interactiva mediante Streamlit.

---

## 🧭 Estructura del Proyecto

```bash
├── app/
│   └── app.py                  # Aplicación principal de Streamlit
│
├── data/
│   ├── fraud.csv               # Dataset por defecto de entrenamiento/EDA
│   ├── fraudTest.csv           # Dataset alternativo de testeo
│   ├── fraud_dataset_encoded.csv
│   ├── shap_global_importance.csv   # Importancia global de variables (SHAP)
│   ├── monitoring_metrics.csv       # Métricas históricas para monitoreo
│   ├── shap_summary_global_bar.png
│   ├── shap_waterfall_local.png
│   └── retrain_warning.txt     # Indicador si se requiere reentrenamiento
│
├── models/
│   ├── rf_model.pkl            # Modelo entrenado principal
│   ├── random_forest_fraud.pkl
│   ├── imputer.pkl             # Imputador guardado
│   └── imputer_median.pkl
│
├── notebooks/
│   ├── eda.ipynb               # Exploración de datos en notebook
│   └── model_pipeline.ipynb   # Pipeline de entrenamiento y validación
│
├── src/
│   ├── data_preprocessing.py  # Ingeniería de variables y limpieza
│   ├── eda_section.py         # Lógica del EDA modularizada
│   ├── model_section.py       # Visualización y métricas de monitoreo
│   ├── model_explainer.py     # Interpretabilidad con SHAP
│   ├── model_training.py      # Entrenamiento de modelo
│   ├── monitoring.py          # Cálculo de drift / degradación
│   ├── outlier_detection.py   # Detección y manejo de outliers
│   └── utils.py               # Simulación de procesos automáticos (cron)
│
├── .env                       # Variables de entorno (paths, config)
└── requirements.txt           # Dependencias del proyecto
```

# ✅ Funcionalidades Principales

## 1. EDA Interactivo en Streamlit
- Gráficos automáticos configurables según el dataset cargado.
- Mapeo dinámico de la variable objetivo.
- Análisis de variables categóricas y numéricas.
- Matriz de correlación automática.
- Soporte para datasets externos.

## 2. Entrenamiento del Modelo
- Modelo por defecto: `RandomForestClassifier`.
- Ingeniería de features clave: diferencias de saldo, errores, ratios.
- Imputación de valores faltantes (guardado del imputador).
- Entrenamiento modularizado y reutilizable.

## 3. Explicabilidad del Modelo
- Cálculo de SHAP values globales.
- Visualización tipo barra y waterfall.
- Interpretación de predicciones y variables más influyentes.

## 4. Monitoreo y Reentrenamiento
- Simulación de entrenamiento y predicción diarios/semanales.
- Monitoreo de métricas: AUC, precision, recall, f1.
- Detección de drift con `Wasserstein distance`.
- Trigger de reentrenamiento automático con alerta visual.

---

# ⚙️ Cómo ejecutar

### 1. Clonar el repo
```bash
git clone https://github.com/juanmartinez1942/fraud-bank.git
cd Fraud-bank
```
## 2. Crear entorno virtual

### Con conda:

```bash
conda create -n fraud_detection python=3.9
conda activate fraud_detection
pip install -r requirements.txt
```

### Con Venv

```bash
python3 -m venv venv
source venv/bin/activate  # En Linux / macOS
venv\Scripts\activate     # En Windows
pip install -r requirements.txt
```

## 3. Ejecutar Streamlit
```bash
streamlit run app/app.py
```

## 💡 Simulaciones tipo CRON

En `utils.py` se implementan procesos automáticos simulados:

| Proceso                   | Frecuencia | Descripción                                      |
|---------------------------|------------|--------------------------------------------------|
| `cron_weekly_train_model()` | Semanal    | Reentrena modelo con nueva data                  |
| `cron_daily_predict()`      | Diaria     | Genera predicciones con el modelo entrenado      |
| `cron_daily_evaluate()`     | Diaria     | Evalúa el modelo, actualiza métricas y drift     |

## 🌱 Variables de Entorno (.env)

```bash
DEFAULT_DATA_PATH=data/fraud.csv  
FRAUD_DATASET=data/fraud_dataset.csv  
MONITORING_METRICS=data/monitoring_metrics.csv  
SHAP_GLOBAL=data/shap_global_importance.csv  
MODEL_PATH=models/rf_model.pkl  
IMPUTER_PATH=models/imputer_median.pkl  
WARNING_FILE=data/retrain_warning.txt  
RANDOM_STATE=42  
SAMPLE_SIZE=200000  
```

## 📊 Tecnologías Utilizadas
	•	Python 3.9+
	•	Pandas, Scikit-learn
	•	Plotly, SHAP
	•	Streamlit
	•	Joblib, dotenv
	•	LazyPredict (testing de modelos)

## 📌 Consideraciones
	•	Dataset desbalanceado → se usó RandomUnderSampler para rebalancear clases.
	•	Validación robusta → incluye evaluación repetida y monitoreo continuo.
	•	App generalizable → cualquier CSV compatible puede usarse sin romperse.