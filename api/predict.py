# Lógica de predicción: carga los modelos y calcula las probabilidades
from tensorflow.keras.models import load_model
from src.config import RUTA_MODELO1, RUTA_MODELO2
from src.data_prep import preparar_cliente
from src.train.utils import nivel_riesgo

# Los modelos se cargan una sola vez, cuando arranca la API
modelo1 = load_model(RUTA_MODELO1)
modelo2 = load_model(RUTA_MODELO2)


def predecir_churn(datos):
    # Modelo 1: ¿el cliente se va o se queda?
    X = preparar_cliente(datos)
    probabilidad = float(modelo1.predict(X, verbose=0)[0][0])
    prediccion = "Se va" if probabilidad > 0.5 else "Se queda"
    return {"probabilidad": round(probabilidad, 4), "prediccion": prediccion}


def predecir_riesgo(datos):
    # Modelo 2: probabilidad de churn y nivel de riesgo
    X = preparar_cliente(datos)
    probabilidad = float(modelo2.predict(X, verbose=0)[0][0])
    return {"probabilidad": round(probabilidad, 4), "nivel": nivel_riesgo(probabilidad)}