# ==============================================================================
# BACKEND & INTERFAZ WEB - TRABAJO FINAL (STREAMLIT)
# ==============================================================================
import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Predicción de Burnout - Dashboard de Gestión Humana",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def cargar_modelo():
    with open("modelo/modelo_burnout.pkl", "rb") as archivo:
        return pickle.load(archivo)

try:
    modelo_desplegado = cargar_modelo()
except FileNotFoundError:
    st.error("⚠️ No se encontró el archivo 'modelo_burnout.pkl' en la carpeta 'modelo/'.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ No se pudo cargar el modelo: {e}")
    st.stop()

st.title("🧠 Sistema de Alerta Temprana: Detección de Burnout en TI")
st.markdown("""
Este sistema predictivo utiliza el modelo optimizado de **Regresión Logística** desarrollado por el equipo analítico.
Permite ingresar los indicadores diarios de un desarrollador para calcular la probabilidad de desgaste crítico (Burnout) y apoyar la toma de decisiones preventivas en Gestión Humana.
""")

st.write("---")

# ------------------------------------------------------------------
# Inicializamos valores por defecto en session_state (solo la 1ra vez)
# ------------------------------------------------------------------
if "daily_work_hours" not in st.session_state:
    st.session_state.daily_work_hours = 8.0
if "sleep_hours" not in st.session_state:
    st.session_state.sleep_hours = 7.0
if "exercise_hours" not in st.session_state:
    st.session_state.exercise_hours = 1.0

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Indicadores de Perfil y Carga Laboral")
    age = st.number_input("Edad del Colaborador:", min_value=18, max_value=70, value=30, step=1)
    experience_years = st.number_input("Años de Experiencia:", min_value=0, max_value=40, value=5, step=1)

    # 1° slider del "presupuesto de 24h": Horas de Trabajo
    daily_work_hours = st.slider(
        "Horas de Trabajo al Día:",
        min_value=0.0, max_value=16.0, step=0.1,
        key="daily_work_hours"
    )

    meetings_per_day = st.slider("Número de Reuniones al Día:", min_value=0, max_value=12, value=3, step=1)
    commits_per_day = st.slider("Número de Commits al Día:", min_value=0, max_value=40, value=10, step=1)
    bugs_per_day = st.slider("Bugs Reportados al Día:", min_value=0, max_value=30, value=5, step=1)

with col2:
    st.subheader("🌱 Indicadores de Bienestar")

    # 2° slider: Horas de Sueño — su tope depende de lo que ya se usó en Trabajo
    horas_restantes_sueno = max(0.0, 24.0 - st.session_state.daily_work_hours)
    max_sueno = round(min(12.0, horas_restantes_sueno), 1)

    # Si el valor guardado ya no cabe en el nuevo presupuesto, lo bajamos automáticamente
    if st.session_state.sleep_hours > max_sueno:
        st.session_state.sleep_hours = max_sueno

    sleep_hours = st.slider(
        "Horas de Sueño Promedio Diarias:",
        min_value=0.0, max_value=max_sueno, step=0.1,
        key="sleep_hours",
        help=f"Máximo disponible: {max_sueno}h (según horas de trabajo ingresadas)"
    )

    # 3° slider: Ejercicio — su tope depende de lo que ya se usó en Trabajo + Sueño
    horas_restantes_ejercicio = max(0.0, 24.0 - st.session_state.daily_work_hours - st.session_state.sleep_hours)
    max_ejercicio = round(min(5.0, horas_restantes_ejercicio), 1)

    if st.session_state.exercise_hours > max_ejercicio:
        st.session_state.exercise_hours = max_ejercicio

    exercise_hours = st.slider(
        "Horas de Ejercicio al Día:",
        min_value=0.0, max_value=max_ejercicio, step=0.1,
        key="exercise_hours",
        help=f"Máximo disponible: {max_ejercicio}h (según trabajo y sueño ingresados)"
    )

    caffeine_intake = st.slider("Unidades de Cafeína al Día:", min_value=0, max_value=10, value=3, step=1,
                                  help="Tazas de café / bebidas con cafeína consumidas en el día")

    # Indicador visual de horas usadas del día
    horas_usadas = daily_work_hours + sleep_hours + exercise_hours
    st.caption(f"🕐 Horas asignadas del día: {horas_usadas:.1f} / 24.0")

st.write("---")

# 4. PROCESAMIENTO Y PREDICCIÓN (BACKEND) — sin cambios
if st.button("🚀 Evaluar Estado del Desarrollador", type="primary"):

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
        prediccion = modelo_desplegado.predict(data_usuario)
        probabilidad = modelo_desplegado.predict_proba(data_usuario)[:, 1][0]
    except Exception as e:
        st.error(f"Error al generar la predicción: {e}")
        st.stop()

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
