# API REST con FastAPI: recibe los datos de un cliente y devuelve las predicciones
from fastapi import FastAPI, HTTPException
from api.schemas import Cliente, RespuestaChurn, RespuestaRiesgo
from api.predict import predecir_churn, predecir_riesgo

app = FastAPI(title="API de predicción de abandono de clientes (Churn)",
              description="Proyecto B - BD-151 Inteligencia Artificial Aplicada",
              version="1.0")


@app.get("/")
def inicio():
    # Para comprobar rápido que la API está encendida
    return {"mensaje": "API funcionando. La documentación está en /docs"}


@app.post("/predict/churn", response_model=RespuestaChurn)
def predict_churn(cliente: Cliente):
    # Modelo 1: ¿el cliente se va o se queda?
    try:
        return predecir_churn(cliente.model_dump())
    except Exception as error:
        raise HTTPException(status_code=500, detail="Error al predecir: " + str(error))


@app.post("/predict/risk_score", response_model=RespuestaRiesgo)
def predict_risk_score(cliente: Cliente):
    # Modelo 2: probabilidad de churn y nivel de riesgo
    try:
        return predecir_riesgo(cliente.model_dump())
    except Exception as error:
        raise HTTPException(status_code=500, detail="Error al predecir: " + str(error))