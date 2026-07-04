# ==============================================================================
# BACKEND & INTERFAZ WEB - TRABAJO FINAL (STREAMLIT)
# ==============================================================================
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

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

# ==============================================================================
# 4. SIDEBAR CON INFORMACIÓN DEL PROYECTO
# ==============================================================================
with st.sidebar:
    st.header("ℹ️ Información del Modelo")
    st.markdown("""
    **Proyecto:** Predicción de Burnout en Desarrolladores TI
    **Algoritmo:** Regresión Logística (Optimizada, class_weight='balanced')
    **Técnica de selección:** RFECV (Recursive Feature Elimination with CV)
    """)
    st.divider()
    st.subheader("📈 Métricas del Modelo (Test)")
    col_a, col_b = st.columns(2)
    col_a.metric("F1-Score", "0.764")
    col_a.metric("Sensibilidad", "0.902")
    col_b.metric("Exactitud", "0.856")
    col_b.metric("Especificidad", "0.840")
    st.divider()
    st.caption(f"Última actualización de sesión: {datetime.now().strftime('%d/%m/%Y')}")
    st.caption("Trabajo Final - Machine Learning - UPC")

# 6. INICIALIZAMOS EL HISTORIAL DE EVALUACIONES DE LA SESIÓN
if "historial" not in st.session_state:
    st.session_state.historial = []

st.title("🧠 Sistema de Alerta Temprana: Detección de Burnout en TI")
st.markdown("""
Este sistema predictivo utiliza el modelo optimizado de **Regresión Logística** desarrollado por el equipo analítico.
Permite ingresar los indicadores diarios de un desarrollador para calcular la probabilidad de desgaste crítico (Burnout) y apoyar la toma de decisiones preventivas en Gestión Humana.
""")

st.write("---")

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

    daily_work_hours = st.slider(
        "Horas de Trabajo al Día:",
        min_value=0.0, max_value=16.0, step=0.1,
        key="daily_work_hours"
    )

    # 3. Validación suave: advertencia si reuniones no cuadran con horas de trabajo
    meetings_per_day = st.slider("Número de Reuniones al Día:", min_value=0, max_value=12, value=3, step=1)
    if meetings_per_day > 0 and daily_work_hours > 0 and (meetings_per_day > daily_work_hours * 1.5):
        st.caption("⚠️ El número de reuniones parece alto respecto a las horas de trabajo ingresadas.")

    commits_per_day = st.slider("Número de Commits al Día:", min_value=0, max_value=40, value=10, step=1)
    bugs_per_day = st.slider("Bugs Reportados al Día:", min_value=0, max_value=30, value=5, step=1)

with col2:
    st.subheader("🌱 Indicadores de Bienestar")

    horas_restantes_sueno = max(0.0, 24.0 - st.session_state.daily_work_hours)
    max_sueno = round(min(12.0, horas_restantes_sueno), 1)
    if st.session_state.sleep_hours > max_sueno:
        st.session_state.sleep_hours = max_sueno

    sleep_hours = st.slider(
        "Horas de Sueño Promedio Diarias:",
        min_value=0.0, max_value=max_sueno, step=0.1,
        key="sleep_hours",
        help=f"Máximo disponible: {max_sueno}h (según horas de trabajo ingresadas)"
    )

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

    horas_usadas = daily_work_hours + sleep_hours + exercise_hours
    st.caption(f"🕐 Horas asignadas del día: {horas_usadas:.1f} / 24.0")

st.write("---")

# ==============================================================================
# 2. FUNCIÓN AUXILIAR: explicar qué variables influyeron más en ESTA predicción
# ==============================================================================
def obtener_factores_influyentes(pipeline, datos_usuario_dict):
    """
    Devuelve una lista de (variable, contribución) usando los coeficientes
    de la Regresión Logística final, aplicados sobre los datos YA escalados
    del usuario. Si algo falla (por estructura del pipeline), retorna None.
    """
    try:
        clasificador = pipeline.named_steps.get('classifier', None)
        preprocesador = pipeline.named_steps.get('preprocessor', None)
        selector = pipeline.named_steps.get('selector', None)

        if clasificador is None or not hasattr(clasificador, 'coef_'):
            return None

        X_proc = preprocesador.transform(pd.DataFrame([datos_usuario_dict])) if preprocesador else None
        if X_proc is None:
            return None

        nombres_features = preprocesador.get_feature_names_out()

        if selector is not None:
            X_proc = selector.transform(X_proc)
            mascara = selector.get_support()
            nombres_features = nombres_features[mascara]

        valores = X_proc[0]
        coeficientes = clasificador.coef_[0]
        contribuciones = valores * coeficientes

        # Limpiamos los prefijos técnicos del ColumnTransformer (ej. "num_minmax__bugs_per_day" -> "bugs_per_day")
        nombres_limpios = [
            nombre.split('__')[-1] if '__' in nombre else nombre
            for nombre in nombres_features
        ]

        df_contrib = pd.DataFrame({
            'Variable': nombres_limpios,
            'Contribución': contribuciones
        }).sort_values('Contribución', ascending=False)

        return df_contrib
    except Exception:
        return None


# 4. PROCESAMIENTO Y PREDICCIÓN (BACKEND)
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
    else:
        st.success(f"✅ **ESTADO ESTABLE:** El modelo clasifica al desarrollador en un rango de **DESGASTE CONTROLADO / SALUDABLE**.")

    # 1. VISUALIZACIÓN DE LA PROBABILIDAD (barra de progreso + métrica)
    col_prob1, col_prob2 = st.columns([1, 2])
    with col_prob1:
        st.metric(label="Probabilidad de Riesgo", value=f"{probabilidad * 100:.2f}%")
    with col_prob2:
        st.write("Nivel de riesgo:")
        st.progress(min(float(probabilidad), 1.0))

    if prediccion[0] == 1:
        st.warning("""
        **Recomendaciones de Intervención Inmediata:**
        * Coordinar una reunión uno a uno con el colaborador para escuchar su situación.
        * Evaluar la reducción de horas de trabajo diarias o la carga de reuniones.
        * Promover hábitos de sueño y ejercicio, y monitorear el consumo de cafeína.
        """)
    else:
        st.info("💡 **Recomendación:** Continuar con el monitoreo periódico de indicadores de carga de trabajo y bienestar.")

    # 2. FACTORES QUE MÁS INFLUYERON EN ESTA PREDICCIÓN
    st.write("---")
    st.subheader("🔍 Factores que más influyeron en este resultado")
    df_factores = obtener_factores_influyentes(modelo_desplegado, data_usuario.iloc[0].to_dict())

    if df_factores is not None and len(df_factores) > 0:
        top_factores = pd.concat([df_factores.head(3), df_factores.tail(3)]).drop_duplicates()
        st.caption("Valores positivos empujan hacia 'Crítico'; valores negativos empujan hacia 'No Crítico'.")
        st.bar_chart(top_factores.set_index('Variable')['Contribución'])
    else:
        st.caption("No fue posible calcular el detalle de factores para esta configuración del pipeline.")

    # 5. BOTÓN DE DESCARGA DE REPORTE
    st.write("---")
    reporte_texto = f"""REPORTE DE EVALUACIÓN - SISTEMA DE ALERTA TEMPRANA DE BURNOUT
Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}

--- DATOS INGRESADOS ---
Edad: {age} años
Años de experiencia: {experience_years}
Horas de trabajo/día: {daily_work_hours}
Horas de sueño/día: {sleep_hours}
Horas de ejercicio/día: {exercise_hours}
Cafeína/día: {caffeine_intake}
Reuniones/día: {meetings_per_day}
Commits/día: {commits_per_day}
Bugs reportados/día: {bugs_per_day}

--- RESULTADO ---
Clasificación: {"BURNOUT CRÍTICO" if prediccion[0] == 1 else "ESTADO ESTABLE"}
Probabilidad de riesgo: {probabilidad * 100:.2f}%
"""
    st.download_button(
        label="📥 Descargar Reporte",
        data=reporte_texto,
        file_name=f"reporte_burnout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

    # 6. GUARDAMOS EN EL HISTORIAL DE LA SESIÓN
    st.session_state.historial.append({
        "Edad": age,
        "Trabajo (h)": daily_work_hours,
        "Sueño (h)": sleep_hours,
        "Resultado": "🚨 Crítico" if prediccion[0] == 1 else "✅ Estable",
        "Probabilidad": f"{probabilidad * 100:.1f}%"
    })

# ==============================================================================
# 6. HISTORIAL DE EVALUACIONES DE LA SESIÓN
# ==============================================================================
if len(st.session_state.historial) > 0:
    st.write("---")
    st.subheader("📋 Historial de Evaluaciones (esta sesión)")
    df_historial = pd.DataFrame(st.session_state.historial)
    st.dataframe(df_historial, use_container_width=True)

    if st.button("🗑️ Limpiar historial"):
        st.session_state.historial = []
        st.rerun()
