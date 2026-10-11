# Página de análisis por lotes: evalúa un grupo de clientes con la API y los segmenta por nivel de riesgo #
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
RAIZ = Path(__file__).resolve().parent.parent.parent   # Carpeta principal del proyecto (tres niveles arriba) #

NIVELES = ["Bajo", "Medio", "Alto", "Crítico"]
COLORES = {"Bajo": "#0ca30c", "Medio": "#fab219", "Alto": "#ec835a", "Crítico": "#d03b3b"}
ACCIONES = {"Bajo": "Ninguna", "Medio": "Seguimiento sin costo",
            "Alto": "Oferta de retención", "Crítico": "Contacto inmediato"}

# Texto en español para los valores que vienen en inglés en el dataset #
TRADUCCION = {"Month-to-month": "Mensual", "One year": "Un año", "Two year": "Dos años", "Yes": "Sí"}

st.set_page_config(page_title="Análisis | Proyecto B", layout="wide")

st.title("Análisis por lotes")
st.write("Se toma un grupo de clientes del dataset, se envía cada uno a la API y se segmentan "
         "según el nivel de riesgo que devuelve el Modelo 2.")

# Datos: la misma limpieza del notebook 02 (clientes sin TotalCharges se eliminan) #
datos = pd.read_csv(RAIZ / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
datos["TotalCharges"] = pd.to_numeric(datos["TotalCharges"], errors="coerce")
datos = datos.dropna()

# Controles #
cantidad = st.slider("Cantidad de clientes a evaluar", min_value=20, max_value=300, value=100, step=20)
evaluar = st.button("Evaluar clientes", type="primary")

# La API recibe los 19 datos del cliente: se quitan el identificador y la respuesta real #
if evaluar:
    muestra = datos.sample(n=cantidad, random_state=101)
    clientes = muestra.drop(columns=["customerID", "Churn"]).to_dict(orient="records")

    barra = st.progress(0.0, text="Consultando la API...")
    probabilidades = []
    niveles = []
    try:
        for i, cliente in enumerate(clientes):
            respuesta = requests.post(API_URL + "/predict/risk_score", json=cliente, timeout=15)
            if respuesta.status_code == 200:
                probabilidades.append(respuesta.json()["probabilidad"])
                niveles.append(respuesta.json()["nivel"])
            else:
                # Si la API rechaza al cliente, queda sin nivel y luego se descarta #
                probabilidades.append(None)
                niveles.append(None)
            barra.progress((i + 1) / cantidad, text=f"Consultando la API... {i + 1} de {cantidad}")
    except requests.exceptions.RequestException:
        st.error("No se pudo conectar con la API. Enciéndala con:  uvicorn api.main:app --reload")
        st.stop()
    barra.empty()

    resultados = muestra[["customerID", "tenure", "Contract", "MonthlyCharges", "Churn"]].copy()
    resultados["Probabilidad"] = probabilidades
    resultados["Nivel"] = niveles
    resultados = resultados.dropna()

    # Se guardan para que no se pierdan cuando la página se vuelve a ejecutar #
    st.session_state["resultados"] = resultados

if "resultados" not in st.session_state:
    st.info("Elija la cantidad de clientes y presione **Evaluar clientes**.")
    st.stop()

resultados = st.session_state["resultados"]

# Indicadores generales #
en_riesgo = resultados[resultados["Nivel"].isin(["Alto", "Crítico"])]

st.header("Resumen")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Clientes evaluados", len(resultados))
col2.metric("En riesgo Alto o Crítico", f"{len(en_riesgo)} ({len(en_riesgo) / len(resultados) * 100:.0f} %)")
col3.metric("Probabilidad promedio", f"{resultados['Probabilidad'].mean() * 100:.1f} %")
col4.metric("Facturación mensual en riesgo", f"${en_riesgo['MonthlyCharges'].sum():,.0f}".replace(",", "."))

# Segmentación por nivel de riesgo #
st.header("Segmentación por nivel de riesgo")

resumen = pd.DataFrame({"Nivel": NIVELES})
resumen["Clientes"] = [(resultados["Nivel"] == nivel).sum() for nivel in NIVELES]
resumen["% del grupo"] = resumen["Clientes"] / len(resultados) * 100
resumen["Churn real (%)"] = [(resultados[resultados["Nivel"] == nivel]["Churn"] == "Yes").mean() * 100
                             for nivel in NIVELES]
resumen["Acción sugerida"] = [ACCIONES[nivel] for nivel in NIVELES]

col1, col2 = st.columns([2, 3])
with col1:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    barras = ax.bar(resumen["Nivel"], resumen["Clientes"], color=[COLORES[nivel] for nivel in NIVELES])
    ax.bar_label(barras)   # la cantidad encima de cada barra
    ax.margins(y=0.15)     # espacio arriba para que el número no choque con el borde
    ax.set_ylabel("Clientes")
    ax.set_title("Clientes por nivel de riesgo")
    st.pyplot(fig)
with col2:
    st.dataframe(resumen, hide_index=True,
                 column_config={"% del grupo": st.column_config.NumberColumn(format="%.1f %%"),
                                "Churn real (%)": st.column_config.NumberColumn(format="%.1f %%")})

st.caption("Churn real: porcentaje de clientes de cada nivel que sí abandonaron según el dataset. "
           "Sirve para ver que los niveles separan bien, pero no es la evaluación del modelo, porque "
           "la mayoría de estos clientes se usaron para entrenar. La evaluación con test está en la página Métricas.")

# Lista de clientes #
st.header("Clientes por nivel")
elegidos = st.multiselect("Niveles a mostrar", NIVELES, default=["Alto", "Crítico"])

lista = resultados[resultados["Nivel"].isin(elegidos)].sort_values("Probabilidad", ascending=False)
lista = lista.replace(TRADUCCION)
lista = lista.rename(columns={"customerID": "Cliente", "tenure": "Antigüedad (meses)", "Contract": "Contrato",
                              "MonthlyCharges": "Cargo mensual ($)", "Churn": "Churn real"})

st.write(f"{len(lista)} clientes en los niveles elegidos, ordenados de mayor a menor probabilidad.")
st.dataframe(lista, hide_index=True,
             column_config={"Probabilidad": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                            "Cargo mensual ($)": st.column_config.NumberColumn(format="%.2f")})

st.download_button("Descargar esta lista (CSV)", lista.to_csv(index=False).encode("utf-8-sig"),
                   file_name="clientes_por_nivel.csv", mime="text/csv")