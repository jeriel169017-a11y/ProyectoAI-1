# Página de métricas: rendimiento de los dos modelos con el conjunto de prueba #
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model

RAIZ = Path(__file__).resolve().parent.parent.parent   # Carpeta principal del proyecto (tres niveles arriba) #

NIVELES = ["Bajo", "Medio", "Alto", "Crítico"]
COLORES = {"Bajo": "#0ca30c", "Medio": "#fab219", "Alto": "#ec835a", "Crítico": "#d03b3b"}

st.set_page_config(page_title="Métricas | Proyecto B", layout="wide")

st.title("Métricas de los modelos")
st.write("Rendimiento de los dos modelos con el conjunto de prueba: clientes que no se usaron para entrenar.")

# Datos de prueba: ya vienen escalados y con dummies desde el notebook 02 #
test = pd.read_csv(RAIZ / "data" / "processed" / "test.csv")
X_test = test.drop('Churn', axis=1).values
y_test = test['Churn'].values

# Predicciones: se calculan una sola vez y Streamlit las guarda #
@st.cache_data
def predecir_test():
    model1 = load_model(RAIZ / "models" / "model1.keras")
    model2 = load_model(RAIZ / "models" / "model2.keras")
    prob1 = model1.predict(X_test, verbose=0).ravel()
    prob2 = model2.predict(X_test, verbose=0).ravel()
    return prob1, prob2


with st.spinner("Cargando los modelos..."):
    prob1, prob2 = predecir_test()

pred1 = (prob1 > 0.5).astype("int32")
pred2 = (prob2 > 0.5).astype("int32")

# Comparación de métricas #
st.header("Comparación")
reporte1 = classification_report(y_test, pred1, output_dict=True)
reporte2 = classification_report(y_test, pred2, output_dict=True)

comparacion = pd.DataFrame({
    "Modelo": ["Modelo 1: decisión (con class_weight)", "Modelo 2: probabilidad (sin class_weight)"],
    "Accuracy": [reporte1["accuracy"], reporte2["accuracy"]],
    "Precisión (se va)": [reporte1["1"]["precision"], reporte2["1"]["precision"]],
    "Recall (se va)": [reporte1["1"]["recall"], reporte2["1"]["recall"]],
    "F1 (se va)": [reporte1["1"]["f1-score"], reporte2["1"]["f1-score"]],
    "AUC": [roc_auc_score(y_test, prob1), roc_auc_score(y_test, prob2)],
})

col1, col2, col3 = st.columns(3)
col1.metric("Clientes en test", f"{len(y_test):,}".replace(",", "."))
col2.metric("Modelo 1: recall", f"{reporte1['1']['recall'] * 100:.0f} %")
col3.metric("Modelo 2: AUC", f"{roc_auc_score(y_test, prob2):.3f}")

st.dataframe(comparacion, hide_index=True,
             column_config={columna: st.column_config.NumberColumn(format="%.3f")
                            for columna in comparacion.columns[1:]})
st.caption("Las dos predicciones usan el corte de 0.5. Recall: de los clientes que se fueron, cuántos detectó el modelo. "
           "Precisión: de los que el modelo marcó como fuga, cuántos sí se fueron. "
           "AUC: qué tan bien ordena a los clientes de mayor a menor riesgo (0.5 es azar, 1 es perfecto).")

# Matrices de confusión #
st.header("Matrices de confusión")
col1, col2 = st.columns(2)
for columna, nombre, pred in [(col1, "Modelo 1", pred1), (col2, "Modelo 2", pred2)]:
    matriz = confusion_matrix(y_test, pred)
    with columna:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=["Se queda", "Se va"], yticklabels=["Se queda", "Se va"], ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(nombre)
        st.pyplot(fig)
        st.write(f"Detectó **{matriz[1][1]}** de los {matriz[1].sum()} clientes que se fueron "
                 f"y dio **{matriz[0][1]}** falsas alarmas.")

# Niveles de riesgo del Modelo 2 #
st.header("Niveles de riesgo en test")
with open(RAIZ / "models" / "umbrales_riesgo.json", encoding="utf-8") as f:
    umbrales = json.load(f)["umbrales"]


def nivel_riesgo(probabilidad):
    # Misma regla que usa la API
    if probabilidad >= umbrales["critico"]:
        return "Crítico"
    if probabilidad >= umbrales["alto"]:
        return "Alto"
    if probabilidad >= umbrales["medio"]:
        return "Medio"
    return "Bajo"


clientes_test = pd.DataFrame({"Nivel": [nivel_riesgo(p) for p in prob2], "Churn": y_test})

resumen = pd.DataFrame({"Nivel": NIVELES})
resumen["Clientes"] = [(clientes_test["Nivel"] == nivel).sum() for nivel in NIVELES]
resumen["% de test"] = resumen["Clientes"] / len(clientes_test) * 100
resumen["Churn real (%)"] = [clientes_test[clientes_test["Nivel"] == nivel]["Churn"].mean() * 100
                             for nivel in NIVELES]

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    barras = ax.bar(resumen["Nivel"], resumen["Churn real (%)"], color=[COLORES[nivel] for nivel in NIVELES])
    ax.bar_label(barras, fmt="%.0f %%")
    ax.margins(y=0.15)
    ax.set_ylabel("Churn real (%)")
    ax.set_title("Clientes que sí se fueron, por nivel")
    st.pyplot(fig)
with col2:
    st.dataframe(resumen, hide_index=True,
                 column_config={"% de test": st.column_config.NumberColumn(format="%.1f %%"),
                                "Churn real (%)": st.column_config.NumberColumn(format="%.1f %%")})
    st.write("Entre más alto el nivel, mayor es el porcentaje de clientes que realmente se fue. "
             "Eso muestra que los umbrales separan bien a los clientes.")

# Curvas de pérdida #
st.header("Curvas de pérdida")
configuracion = st.radio("Configuración", ["Sin Dropout", "Con Dropout"], horizontal=True)
archivo = "sin_dropout" if configuracion == "Sin Dropout" else "con_dropout"

col1, col2 = st.columns(2)
for columna, numero in [(col1, 1), (col2, 2)]:
    historial = pd.read_csv(RAIZ / "models" / f"historial_modelo{numero}_{archivo}.csv")
    with columna:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(historial["loss"], label="Entrenamiento", color="#2a78d6")
        ax.plot(historial["val_loss"], label="Validación", color="#eb6834")
        ax.set_xlabel("Época")
        ax.set_ylabel("Pérdida")
        ax.set_title(f"Modelo {numero} ({configuracion.lower()})")
        ax.legend()
        st.pyplot(fig)

st.caption("Configuración final: el Modelo 1 se guardó con Dropout y el Modelo 2 sin Dropout. "
           "La comparación completa de las cuatro configuraciones está en el notebook 05.")