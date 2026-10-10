# Rutas y parámetros globales del proyecto
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent   # carpeta principal del proyecto

RUTA_RAW = RAIZ / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RUTA_TRAIN = RAIZ / "data" / "processed" / "train.csv"
RUTA_TEST = RAIZ / "data" / "processed" / "test.csv"

RUTA_MODELO1 = RAIZ / "models" / "model1.keras"
RUTA_MODELO2 = RAIZ / "models" / "model2.keras"
RUTA_SCALER = RAIZ / "models" / "scaler.pkl"
RUTA_COLUMNAS = RAIZ / "models" / "columnas.pkl"
RUTA_UMBRALES = RAIZ / "models" / "umbrales_riesgo.json"

SEMILLA = 101
SERVICIOS_EXTRA = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]

# Umbrales y niveles de riesgo, los mismos que se definieron en el notebook 04
with open(RUTA_UMBRALES, encoding="utf-8") as f:
    datos = json.load(f)
UMBRALES = datos["umbrales"]
NIVELES = datos["niveles"]