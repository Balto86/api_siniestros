# =========================================================
# APP WEB - PREDICCIÓN DE SINIESTROS DE TRÁNSITO
# =========================================================
# Autor: David Fernando Cargua León
# Descripción: Interfaz web que consume la API de predicción
# =========================================================

import streamlit as st
import requests

# =========================================================
# CONFIGURACIÓN DE LA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Predicción de Siniestros",
    page_icon="🚦",
    layout="wide"
)

# URL de la API
API_URL = "http://localhost:5000/predecir"

# =========================================================
# TÍTULO Y DESCRIPCIÓN
# =========================================================
st.title("🚦 Sistema de Predicción de Siniestros de Tránsito")
st.markdown("""
Esta aplicación utiliza un modelo de **Machine Learning (Random Forest)** 
para predecir la **causa más probable** de un siniestro de tránsito 
basándose en variables temporales, geográficas y de vehículos.

**Precisión del modelo:** 92.30% | **ROC-AUC:** 0.9793
""")

st.markdown("---")

# =========================================================
# FORMULARIO DE ENTRADA
# =========================================================
st.header("📋 Datos del Siniestro")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("⏰ Temporales")
    hora = st.slider("Hora del día (0-23)", 0, 23, 14)
    dia = st.selectbox(
        "Día de la semana",
        ["LUNES", "MARTES", "MIÉRCOLES", "JUEVES", "VIERNES", "SÁBADO", "DOMINGO"]
    )
    feriado = st.selectbox("¿Es feriado?", ["NO", "SI"])

with col2:
    st.subheader("📍 Geográficos")
    zona = st.selectbox("Zona", ["URBANA", "RURAL"])
    provincia = st.selectbox(
        "Provincia",
        ["PICHINCHA", "GUAYAS", "AZUAY", "MANABI", "TUNGURAHUA", "OTRA"]
    )

with col3:
    st.subheader("🚗 Vehículos y Severidad")
    tipo_siniestro = st.selectbox(
        "Tipo de siniestro",
        ["CHOQUE LATERAL", "CHOQUE FRONTAL", "CHOQUE POR ALCANCE", 
         "ATROPELLO", "VOLCAMIENTO", "CAIDA DE PASAJERO", "OTRO"]
    )
    lesionados = st.number_input("Número de lesionados", 0, 50, 1)
    fallecidos = st.number_input("Número de fallecidos", 0, 20, 0)

col4, col5 = st.columns(2)

with col4:
    tiene_moto = st.checkbox("¿Hubo motocicleta involucrada?")
    tiene_auto = st.checkbox("¿Hubo automóvil involucrado?", value=True)

with col5:
    tiene_bus = st.checkbox("¿Hubo bus involucrado?")
    tiene_camion = st.checkbox("¿Hubo camión involucrado?")

st.markdown("---")

# =========================================================
# BOTÓN DE PREDICCIÓN
# =========================================================
if st.button("🔮 Predecir Causa", type="primary", use_container_width=True):
    
    datos = {
        "HORA_NUM": hora,
        "DIA_1": dia,
        "ES_FERIADO": feriado,
        "ZONA": zona,
        "PROVINCIA": provincia,
        "TIPO_DE_SINIESTRO": tipo_siniestro,
        "LESIONADOS": lesionados,
        "FALLECIDOS": fallecidos,
        "TIENE_MOTOCICLETA": 1 if tiene_moto else 0,
        "TIENE_AUTOMOVIL": 1 if tiene_auto else 0
    }
    
    try:
        with st.spinner("Consultando el modelo..."):
            respuesta = requests.post(API_URL, json=datos, timeout=10)
        
        if respuesta.status_code == 200:
            resultado = respuesta.json()
            
            st.markdown("---")
            st.header("🎯 Resultado de la Predicción")
            
            prediccion = resultado["prediccion"]
            st.success(f"### Causa más probable: **{prediccion}**")
            
            confianza = resultado["probabilidades"][prediccion]
            st.metric("Nivel de confianza", f"{confianza * 100:.2f}%")
            
            st.subheader("📊 Probabilidades por causa")
            
            probs = resultado["probabilidades"]
            probs_ordenadas = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
            
            for causa, prob in probs_ordenadas.items():
                st.write(f"**{causa}**")
                st.progress(prob)
                st.caption(f"{prob * 100:.2f}%")
            
            st.markdown("---")
            if confianza > 0.7:
                st.info("✅ El modelo tiene **alta confianza** en esta predicción.")
            elif confianza > 0.5:
                st.warning("⚠️ El modelo tiene **confianza moderada**.")
            else:
                st.warning("⚠️ El modelo tiene **baja confianza**.")
        
        else:
            st.error(f"❌ Error en la API: {respuesta.status_code}")
            st.json(respuesta.json())
    
    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar a la API. Asegúrate de que `python app.py` esté corriendo.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("🎓 Proyecto de Minería de Datos | Random Forest | API Flask + Streamlit")