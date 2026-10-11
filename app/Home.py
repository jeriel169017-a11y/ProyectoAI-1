# Página principal del dashboard: presenta el sistema y revisa que la API esté encendida #
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
RAIZ = Path(__file__).resolve().parent.parent   # Carpeta principal del proyecto #

st.set_page_config(page_title= "Churn Proyecto B", layout= "wide")

st.title("Sistema de Predicción de Abandono de Clientes")
st.caption("Proyecto B - BD-151 Inteligencia Artificial Aplicada - Colegio Universitario de Cartago")

# Estado de la API: las páginas de predicción la necesitan encendida #
try:
    respuesta = requests.get(API_URL + "/", timeout= 3)
    api_encendida = respuesta.status_code == 200
except requests.exceptions.RequestException:
    api_encendida = False

if api_encendida:
    st.success("La API está encendida y lista para predecir.")
else:
    st.error("La API no responde. Enciéndala desde la carpeta del proyecto con : uvicorn api.main:app --reload")

# Resumen del dataset #
st.header("El problema")
st.write("Una empresa de telecomunicaciones quiere saber qué clientes tienen más probabilidad de abandonar "
         "el servicio, para atenderlos antes de que se vayan.")

datos = pd.read_csv(RAIZ / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
tasa_churn = (datos["Churn"] == "Yes").mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Clientes en el dataset", f"{len(datos):,}".replace(",", "."))
col2.metric("Tasa de churn", f"{tasa_churn:.1f} %")
col3.metric("Cargo mensual promedio", f"${datos['MonthlyCharges'].mean():.2f}")
col4.metric("Antigüedad promedio", f"{datos['tenure'].mean():.0f} meses")

# Integración de los dos modelos #
st.header("Los modelos")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Modelo 1: ¿se va o se queda?")
    st.write("Red neuronal de clasificación binaria. Se entrena con `class_weight` para no dejar escapar "
             "a los clientes que se van. Responde **Se va** o **Se queda**.")
with col2:
    st.subheader("Modelo 2: ¿qué tan probable es?")
    st.write("Red neuronal que estima la probabilidad de churn, de 0 a 1. Con esa probabilidad se ubica "
             "al cliente en un nivel de riesgo.")

# Niveles de riesgo: se leen del mismo archivo que usa la API #
st.header("Niveles de riesgo")
with open(RAIZ / "models" / "umbrales_riesgo.json", encoding="utf-8") as f:
    umbrales = json.load(f)["umbrales"]

niveles = pd.DataFrame({
    "Nivel": ["Bajo", "Medio", "Alto", "Crítico"],
    "Probabilidad de churn": [f"Menor a {umbrales['medio']:.2f}",
                              f"De {umbrales['medio']:.2f} a {umbrales['alto']:.2f}",
                              f"De {umbrales['alto']:.2f} a {umbrales['critico']:.2f}",
                              f"{umbrales['critico']:.2f} o más"],
    "Acción sugerida": ["Ninguna", "Seguimiento sin costo", "Oferta de retención", "Contacto inmediato"],
})
st.dataframe(niveles, hide_index=True)

# Guía de páginas #
st.header("Páginas del dashboard")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Predicción")
    st.write("Se ingresan los datos de un cliente y el sistema responde si se va y con qué nivel de riesgo.")
with col2:
    st.subheader("2. Análisis")
    st.write("Se evalúa un grupo de clientes y se segmentan por nivel de riesgo.")
with col3:
    st.subheader("3. Métricas")
    st.write("Rendimiento de los dos modelos con el conjunto de prueba.")
st.caption("Use el menú de la izquierda para cambiar de página.")