# Prepara los datos de un cliente nuevo igual que hacemos en el notebook2
import pandas as pd
import joblib
from src.config import RUTA_SCALER, RUTA_COLUMNAS, SERVICIOS_EXTRA


def limpiar(df):
    df = df.copy()
    # TotalCharges a número; si viene vacío se pone 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    # Unificar "No internet service" y "No phone service" en "No"
    df = df.replace({"No internet service": "No", "No phone service": "No"})
    # Feature engineering: cuántos servicios extra tiene el cliente
    df["num_servicios"] = (df[SERVICIOS_EXTRA] == "Yes").sum(axis=1)
    return df


def preparar_cliente(datos):
    df = limpiar(pd.DataFrame([datos]))   # una sola fila con los datos del cliente
    df = pd.get_dummies(df, dtype=int)    # texto a 0 y 1
    columnas = joblib.load(RUTA_COLUMNAS)
    df = df.reindex(columns=columnas, fill_value=0)   # mismas columnas que en el entrenamiento
    scaler = joblib.load(RUTA_SCALER)
    return scaler.transform(df)           # escalar entre 0 y 1