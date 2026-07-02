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

# 3. INTERFAZ DE USUARIO (UI)
st.title("🧠 Sistema de Alerta Temprana: Detección de Burnout en TI")
st.markdown("""
Este sistema predictivo utiliza el modelo optimizado de **Regresión Logística** desarrollado por el equipo analítico. 
Permite ingresar los indicadores de rendimiento y bienestar de un desarrollador para calcular la probabilidad de desgaste crítico (Burnout) y apoyar la toma de decisiones preventivas en Gestión Humana.
""")

st.write("---")

# Creamos dos columnas para organizar los campos de ingreso de datos de forma limpia
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Indicadores de Carga Laboral y Rendimiento")
    horas_trabajo = st.slider("Horas de Trabajo a la Semana:", min_value=20, max_value=80, value=40, step=1)
    commits = st.slider("Número de Commits Semanales:", min_value=0, max_value=150, value=30, step=1)
    bugs = st.slider("Bugs Reportados en la Semana:", min_value=0, max_value=50, value=5, step=1)
    proyectos = st.selectbox("Proyectos Asignados simultáneamente:", options=[1, 2, 3, 4, 5, 6], index=1)
    anios_experiencia = st.number_input("Años de Experiencia del Colaborador:", min_value=0, max_value=40, value=3, step=1)

with col2:
    st.subheader("🌱 Indicadores de Bienestar y Entorno")
    horas_sueno = st.slider("Horas de Sueño Promedio Diarias:", min_value=3, max_value=12, value=7, step=1)
    satisfaccion = st.radio("Nivel de Satisfacción Laboral:", options=[1, 2, 3, 4, 5], index=3, horizontal=True, help="1: Muy Insatisfecho, 5: Muy Satisfecho")
    soporte = st.radio("Nivel de Soporte del Equipo / Líder:", options=[1, 2, 3, 4, 5], index=3, horizontal=True, help="1: Sin soporte, 5: Excelente soporte")
    vacaciones = st.number_input("Días de Vacaciones Tomados en el año:", min_value=0, max_value=30, value=10, step=1)

st.write("---")

# 4. PROCESAMIENTO Y PREDICCIÓN (BACKEND)
if st.button("🚀 Evaluar Estado del Desarrollador", type="primary"):
    
    # Construimos el DataFrame exactamente con los nombres de las columnas originales que espera tu Pipeline
    data_usuario = pd.DataFrame([{
        'Horas_Trabajo_Semana': horas_trabajo,
        'Horas_Sueno_Promedio': horas_sueno,
        'Commits_Semanales': commits,
        'Bugs_Reportados': bugs,
        'Proyectos_Asignados': proyectos,
        'Satisfaccion_Laboral': satisfaccion,
        'Soporte_Equipo': soporte,
        'Dias_Vacaciones_Tomados': vacaciones,
        'Anios_Experiencia': anios_experiencia
    }])
    
    # El pipeline empaquetado en el pickle ejecuta el preprocesamiento, selecciona las 7 mejores variables y predice
    prediccion = modelo_desplegado.predict(data_usuario)
    probabilidad = modelo_desplegado.predict_proba(data_usuario)[:, 1][0]
    
    # 5. DESPLIEGUE DE RESULTADOS EN LA INTERFAZ
    st.subheader("🎯 Resultado del Diagnóstico Predictivo")
    
    if prediccion[0] == 1:
        st.error(f"🚨 **ALERTA CRÍTICA:** El modelo clasifica al desarrollador en riesgo de **BURNOUT CRÍTICO**.")
        st.metric(label="Probabilidad de Riesgo", value=f"{probabilidad * 100:.2f}%")
        st.warning("""
        **Recomendaciones de Intervención Inmediata:**
        * Coordinar una reunión uno a uno con el colaborador para escuchar su situación.
        * Evaluar la reducción de la cantidad de proyectos asignados o redistribuir los commits/tareas de la semana.
        * Promover de forma obligatoria el uso de días de vacaciones acumulados.
        """)
    else:
        st.success(f"✅ **ESTADO ESTABLE:** El modelo clasifica al desarrollador en un rango de **DESGASTE CONTROLADO / SALUDABLE**.")
        st.metric(label="Probabilidad de Riesgo", value=f"{probabilidad * 100:.2f}%")
        st.info("💡 **Recomendación:** Continuar con el monitoreo mensual de indicadores de carga de trabajo y asegurar que se mantengan los niveles de soporte del equipo.")