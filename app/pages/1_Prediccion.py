# Página de predicción individual: se llenan los datos de un cliente y se consulta la API #
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

# Texto en español que se muestra en la pantalla para cada valor que espera la API #
ETIQUETAS = {
    "Yes": "Sí", "No": "No", "Female": "Femenino", "Male": "Masculino",
    "DSL": "DSL", "Fiber optic": "Fibra óptica",
    "Month-to-month": "Mensual", "One year": "Un año", "Two year": "Dos años",
    "Electronic check": "Cheque electrónico", "Mailed check": "Cheque por correo",
    "Bank transfer (automatic)": "Transferencia bancaria (automática)",
    "Credit card (automatic)": "Tarjeta de crédito (automática)",
}

ACCIONES = {"Bajo": "Ninguna", "Medio": "Seguimiento sin costo",
            "Alto": "Oferta de retención", "Crítico": "Contacto inmediato"}

st.set_page_config(page_title="Predicción | Proyecto B", layout="wide")

st.title("Predicción individual")
st.write("Complete los datos del cliente y presione **Predecir**. El dashboard envía los datos a la API "
         "y muestra la respuesta de los dos modelos.")

# Formulario: los datos se envían todos juntos al presionar el botón #
with st.form("formulario_cliente"):
    st.subheader("Datos del cliente")
    col1, col2, col3, col4 = st.columns(4)
    gender = col1.selectbox("Género", ["Female", "Male"], format_func=ETIQUETAS.get)
    senior = col2.selectbox("Adulto mayor", ["No", "Yes"], format_func=ETIQUETAS.get)
    partner = col3.selectbox("Tiene pareja", ["No", "Yes"], format_func=ETIQUETAS.get)
    dependents = col4.selectbox("Tiene dependientes", ["No", "Yes"], format_func=ETIQUETAS.get)

    st.subheader("Cuenta")
    col1, col2, col3, col4 = st.columns(4)
    tenure = col1.number_input("Antigüedad (meses)", min_value=1, max_value=72, value=12)
    contract = col2.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"],
                              format_func=ETIQUETAS.get)
    payment = col3.selectbox("Método de pago", ["Electronic check", "Mailed check",
                                                "Bank transfer (automatic)", "Credit card (automatic)"],
                             format_func=ETIQUETAS.get)
    paperless = col4.selectbox("Factura digital", ["Yes", "No"], format_func=ETIQUETAS.get)

    col1, col2 = st.columns(2)
    monthly = col1.number_input("Cargo mensual ($)", min_value=0.0, value=70.0, step=1.0)
    total = col2.number_input("Total pagado hasta hoy ($)", min_value=0.0, value=840.0, step=10.0)

    st.subheader("Servicios")
    col1, col2, col3 = st.columns(3)
    phone = col1.selectbox("Servicio de teléfono", ["Yes", "No"], format_func=ETIQUETAS.get)
    multiple = col2.selectbox("Varias líneas", ["No", "Yes"], format_func=ETIQUETAS.get)
    internet = col3.selectbox("Internet", ["Fiber optic", "DSL", "No"], format_func=ETIQUETAS.get)

    col1, col2, col3 = st.columns(3)
    security = col1.selectbox("Seguridad en línea", ["No", "Yes"], format_func=ETIQUETAS.get)
    backup = col2.selectbox("Respaldo en línea", ["No", "Yes"], format_func=ETIQUETAS.get)
    protection = col3.selectbox("Protección del equipo", ["No", "Yes"], format_func=ETIQUETAS.get)
    support = col1.selectbox("Soporte técnico", ["No", "Yes"], format_func=ETIQUETAS.get)
    tv = col2.selectbox("Televisión por internet", ["No", "Yes"], format_func=ETIQUETAS.get)
    movies = col3.selectbox("Películas por internet", ["No", "Yes"], format_func=ETIQUETAS.get)

    enviado = st.form_submit_button("Predecir", type="primary")

# Un cliente sin teléfono o sin internet no puede tener los servicios que dependen de ellos #
if enviado:
    if phone == "No":
        multiple = "No phone service"
    if internet == "No":
        security = backup = protection = support = tv = movies = "No internet service"
        st.info("El cliente no tiene internet, así que los seis servicios adicionales se enviaron como no contratados.")

# Los 19 datos, con los nombres que espera la API #
    cliente = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents,
        "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "tenure": tenure, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }

# Consulta a los dos endpoints #
    try:
        respuesta_churn = requests.post(API_URL + "/predict/churn", json=cliente, timeout=15)
        respuesta_riesgo = requests.post(API_URL + "/predict/risk_score", json=cliente, timeout=15)
    except requests.exceptions.RequestException:
        st.error("No se pudo conectar con la API. Enciéndala con:  uvicorn api.main:app --reload")
        st.stop()

    if respuesta_churn.status_code != 200 or respuesta_riesgo.status_code != 200:
        st.error("La API no aceptó los datos: " + str(respuesta_churn.json().get("detail")))
        st.stop()

    churn = respuesta_churn.json()
    riesgo = respuesta_riesgo.json()

# Resultado #
    st.header("Resultado")
    col1, col2, col3 = st.columns(3)
    col1.metric("Modelo 1: decisión", churn["prediccion"])
    col2.metric("Modelo 2: probabilidad de churn", f"{riesgo['probabilidad'] * 100:.1f} %")
    col3.metric("Nivel de riesgo", riesgo["nivel"])

    st.progress(riesgo["probabilidad"], text="Probabilidad de churn según el Modelo 2")

    mensaje = f"Nivel de riesgo {riesgo['nivel']}. Acción sugerida: {ACCIONES[riesgo['nivel']]}."
    if riesgo["nivel"] == "Crítico":
        st.error(mensaje)
    elif riesgo["nivel"] == "Alto":
        st.warning(mensaje)
    elif riesgo["nivel"] == "Medio":
        st.info(mensaje)
    else:
        st.success(mensaje)

    st.caption("El Modelo 1 se entrenó con class_weight, así que marca como fuga a más clientes. "
               "La probabilidad y el nivel de riesgo vienen del Modelo 2.")

    with st.expander("Ver los datos enviados a la API"):
        st.json(cliente)