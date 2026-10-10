# Funciones compartidas para evaluar y graficar los modelos
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


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
    print(classification_report(y_real, y_pred, target_names=["No se va", "Se va"]))
    ConfusionMatrixDisplay.from_predictions(y_real, y_pred, display_labels=["No se va", "Se va"])
    plt.title(titulo)
    plt.show()