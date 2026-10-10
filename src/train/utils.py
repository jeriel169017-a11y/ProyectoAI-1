# Funciones compartidas para evaluar y graficar los modelos
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from src.config import UMBRALES


def graficar_perdida(historial, titulo):
    # Curva de pérdida de entrenamiento y validación
    plt.plot(historial.history["loss"], label="entrenamiento")
    plt.plot(historial.history["val_loss"], label="validación")
    plt.title(titulo)
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.show()


def evaluar_clasificacion(y_real, y_pred, titulo):
    # Reporte de métricas y matriz de confusión
    print(classification_report(y_real, y_pred, target_names=["Se queda", "Se va"]))
    ConfusionMatrixDisplay.from_predictions(y_real, y_pred, display_labels=["Se queda", "Se va"])
    plt.title(titulo)
    plt.show()


def nivel_riesgo(probabilidad):
    # Convierte la probabilidad de churn en un nivel de riesgo
    if probabilidad >= UMBRALES["critico"]:
        return "Crítico"
    if probabilidad >= UMBRALES["alto"]:
        return "Alto"
    if probabilidad >= UMBRALES["medio"]:
        return "Medio"
    return "Bajo"