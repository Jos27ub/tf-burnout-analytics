# ==============================================================================
# BACKEND & INTERFAZ WEB - TRABAJO FINAL (STREAMLIT)
# ==============================================================================
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. CONFIGURACIÓN DE LA PÁGINA (Identidad del proyecto)
st.set_page_config(
    page_title="Predicción de Burnout - Dashboard de Gestión Humana",
    page_icon="🧠",
    layout="wide"
)

# 2. CARGA DEL MODELO DESDE EL ARCHIVO PICKLE
@st.cache_resource # Mantiene el modelo en memoria para que cargue instantáneamente
def cargar_modelo():
    with open("modelo/modelo_burnout.pkl", "rb") as archivo:
        return pickle.load(archivo)

try:
    modelo_desplegado = cargar_modelo()
except FileNotFoundError:
    st.error("⚠️ No se encontró el archivo 'modelo_burnout.pkl' en la carpeta 'modelo/'. Asegúrate de crear la carpeta y colocar el archivo ahí.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ No se pudo cargar el modelo: {e}")
    st.stop()

# 3. INTERFAZ DE USUARIO (UI)
st.title("🧠 Sistema de Alerta Temprana: Detección de Burnout en TI")
st.markdown("""
Este sistema predictivo utiliza el modelo optimizado de **Regresión Logística** desarrollado por el equipo analítico.
Permite ingresar los indicadores diarios de un desarrollador para calcular la probabilidad de desgaste crítico (Burnout) y apoyar la toma de decisiones preventivas en Gestión Humana.
""")

st.write("---")

# Creamos dos columnas para organizar los campos de ingreso de datos de forma limpia
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Indicadores de Perfil y Carga Laboral")
    age = st.number_input("Edad del Colaborador:", min_value=18, max_value=70, value=30, step=1)
    experience_years = st.number_input("Años de Experiencia:", min_value=0, max_value=40, value=5, step=1)
    daily_work_hours = st.slider("Horas de Trabajo al Día:", min_value=0.0, max_value=16.0, value=8.0, step=0.1)
    meetings_per_day = st.slider("Número de Reuniones al Día:", min_value=0, max_value=12, value=3, step=1)
    commits_per_day = st.slider("Número de Commits al Día:", min_value=0, max_value=40, value=10, step=1)
    bugs_per_day = st.slider("Bugs Reportados al Día:", min_value=0, max_value=30, value=5, step=1)

with col2:
    st.subheader("🌱 Indicadores de Bienestar")
    sleep_hours = st.slider("Horas de Sueño Promedio Diarias:", min_value=0.0, max_value=12.0, value=7.0, step=0.1)
    exercise_hours = st.slider("Horas de Ejercicio al Día:", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    caffeine_intake = st.slider("Unidades de Cafeína al Día:", min_value=0, max_value=10, value=3, step=1, help="Tazas de café / bebidas con cafeína consumidas en el día")

st.write("---")

# 4. PROCESAMIENTO Y PREDICCIÓN (BACKEND)
if st.button("🚀 Evaluar Estado del Desarrollador", type="primary"):

    # Construimos el DataFrame exactamente con los nombres de las columnas que espera el Pipeline entrenado
    data_usuario = pd.DataFrame([{
        'age': age,
        'experience_years': experience_years,
        'daily_work_hours': daily_work_hours,
        'sleep_hours': sleep_hours,
        'exercise_hours': exercise_hours,
        'caffeine_intake': caffeine_intake,
        'meetings_per_day': meetings_per_day,
        'commits_per_day': commits_per_day,
        'bugs_per_day': bugs_per_day
    }])

    try:
        # El pipeline empaquetado en el pickle ejecuta el preprocesamiento, selecciona las mejores variables y predice
        prediccion = modelo_desplegado.predict(data_usuario)
        probabilidad = modelo_desplegado.predict_proba(data_usuario)[:, 1][0]
    except Exception as e:
        st.error(f"Error al generar la predicción: {e}")
        st.stop()

    # 5. DESPLIEGUE DE RESULTADOS EN LA INTERFAZ
    st.subheader("🎯 Resultado del Diagnóstico Predictivo")

    if prediccion[0] == 1:
        st.error(f"🚨 **ALERTA CRÍTICA:** El modelo clasifica al desarrollador en riesgo de **BURNOUT CRÍTICO**.")
        st.metric(label="Probabilidad de Riesgo", value=f"{probabilidad * 100:.2f}%")
        st.warning("""
        **Recomendaciones de Intervención Inmediata:**
        * Coordinar una reunión uno a uno con el colaborador para escuchar su situación.
        * Evaluar la reducción de horas de trabajo diarias o la carga de reuniones.
        * Promover hábitos de sueño y ejercicio, y monitorear el consumo de cafeína.
        """)
    else:
        st.success(f"✅ **ESTADO ESTABLE:** El modelo clasifica al desarrollador en un rango de **DESGASTE CONTROLADO / SALUDABLE**.")
        st.metric(label="Probabilidad de Riesgo", value=f"{probabilidad * 100:.2f}%")
        st.info("💡 **Recomendación:** Continuar con el monitoreo periódico de indicadores de carga de trabajo y bienestar.")
