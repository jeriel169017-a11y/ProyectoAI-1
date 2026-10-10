# Rutas y parámetros globales
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent   # carpeta principal de nuestro proyecto

RUTA_RAW = RAIZ / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RUTA_TRAIN = RAIZ / "data" / "processed" / "train.csv"
RUTA_TEST = RAIZ / "data" / "processed" / "test.csv"

RUTA_MODELO1 = RAIZ / "models" / "model1.keras"
RUTA_MODELO2 = RAIZ / "models" / "model2.keras"
RUTA_SCALER = RAIZ / "models" / "scaler.pkl"
RUTA_COLUMNAS = RAIZ / "models" / "columnas.pkl"

SEMILLA = 101
SERVICIOS_EXTRA = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]