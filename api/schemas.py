# Define los datos que recibe y devuelve la API, y valida que vengan correctos
from typing import Literal
from pydantic import BaseModel, Field

# Atajo para las columnas que solo aceptan Yes o No
SiNo = Literal["Yes", "No"]


class Cliente(BaseModel):
    # Datos del cliente
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: SiNo
    Dependents: SiNo

    # Servicios contratados
    PhoneService: SiNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]

    # Información de la cuenta
    tenure: int = Field(ge=1, le=72)   # meses con la empresa
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: SiNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    # Ejemplo que aparece prellenado en la documentación automática (/docs)
    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "tenure": 5, "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 420.0}}}


class RespuestaChurn(BaseModel):
    # Lo que devuelve /predict/churn
    probabilidad: float
    prediccion: str


class RespuestaRiesgo(BaseModel):
    # Lo que devuelve /predict/risk_score
    probabilidad: float
    nivel: str