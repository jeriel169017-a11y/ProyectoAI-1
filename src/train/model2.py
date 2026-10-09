"""
Entrenamiento del Modelo 2: scoring de riesgo de churn.

Reproduce la configuración elegida en notebooks/04_ANN_Modelo2.ipynb
(3 capas ocultas 48-24-12, sin Dropout y sin class_weight) y guarda:
    models/model2.keras
    models/umbrales_riesgo.json

Uso, desde la carpeta raíz del proyecto:
    python -m src.train.model2
"""

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score

RAIZ = Path(__file__).resolve().parents[2]
RUTA_TRAIN = RAIZ / "data" / "processed" / "train.csv"
RUTA_TEST = RAIZ / "data" / "processed" / "test.csv"
RUTA_MODELO = RAIZ / "models" / "model2.keras"
RUTA_UMBRALES = RAIZ / "models" / "umbrales_riesgo.json"

# Configuración elegida en el notebook 04 #
CAPAS_OCULTAS = [48, 24, 12]
DROPOUT = 0.0  # con 5 semillas, sin Dropout dio una val_loss apenas mejor (notebook 04, sección 3.4)
EPOCHS = 600
PATIENCE = 25
VALIDATION_SPLIT = 0.2
SEMILLA = 101

# Umbrales de riesgo justificados en el notebook 04 (sección 4) #
UMBRALES = {"medio": 0.1, "alto": 0.3, "critico": 0.6}
NIVELES = ["Bajo", "Medio", "Alto", "Crítico"]


def cargar_datos():
    train = pd.read_csv(RUTA_TRAIN)
    test = pd.read_csv(RUTA_TEST)
    X_train = train.drop(columns="Churn").values.astype("float32")
    y_train = train["Churn"].values
    X_test = test.drop(columns="Churn").values.astype("float32")
    y_test = test["Churn"].values
    return X_train, y_train, X_test, y_test


def crear_modelo(num_entradas):
    tf.keras.utils.set_random_seed(SEMILLA)
    model = Sequential()
    model.add(Input(shape=(num_entradas,)))
    for neuronas in CAPAS_OCULTAS:
        model.add(Dense(units=neuronas, activation="relu"))
        if DROPOUT > 0:
            model.add(Dropout(DROPOUT))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def nivel_riesgo(probabilidad):
    if probabilidad >= UMBRALES["critico"]:
        return "Crítico"
    if probabilidad >= UMBRALES["alto"]:
        return "Alto"
    if probabilidad >= UMBRALES["medio"]:
        return "Medio"
    return "Bajo"


def entrenar():
    X_train, y_train, X_test, y_test = cargar_datos()

    # Sin class_weight: inflaría las probabilidades y este modelo debe dar probabilidades reales #
    model = crear_modelo(X_train.shape[1])
    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1,
                               patience=PATIENCE, restore_best_weights=True)
    model.fit(x=X_train, y=y_train, epochs=EPOCHS,
              validation_split=VALIDATION_SPLIT,
              callbacks=[early_stop], verbose=0)

    # Evaluación final con test #
    prob_test = model.predict(X_test, verbose=0).ravel()
    print(classification_report(y_test, (prob_test > 0.5).astype("int32"),
                                target_names=["Se queda", "Se va"]))
    print("AUC test:", round(roc_auc_score(y_test, prob_test), 4))
    print("Probabilidad media predicha:", round(prob_test.mean(), 3), "| churn real:", round(y_test.mean(), 3))
    print(pd.Series([nivel_riesgo(p) for p in prob_test]).value_counts().reindex(NIVELES))

    model.save(RUTA_MODELO)
    with open(RUTA_UMBRALES, "w", encoding="utf-8") as f:
        json.dump({"umbrales": UMBRALES, "niveles": NIVELES}, f, indent=2)  # en ASCII: se lee bien con cualquier codificación
    print("Guardados:", RUTA_MODELO.name, "y", RUTA_UMBRALES.name)


if __name__ == "__main__":
    entrenar()
