# Entrenamiento del modelo 1: clasificación binaria (¿el cliente abandona?)
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src.config import RUTA_TRAIN, RUTA_MODELO1, SEMILLA


def cargar_train():
    # Lee train.csv y separa las entradas (X) del objetivo (y)
    train = pd.read_csv(RUTA_TRAIN)
    X = train.drop("Churn", axis=1).values
    y = train["Churn"].values
    return X, y


def calcular_pesos(y):
    # La clase con menos clientes (los que se van) recibe más peso
    total = len(y)
    cantidad_0 = int((y == 0).sum())
    cantidad_1 = int((y == 1).sum())
    return {0: total / (2 * cantidad_0), 1: total / (2 * cantidad_1)}


def crear_modelo(num_entradas, dropout):
    # Red con 2 capas ocultas y una neurona de salida (probabilidad de abandono)
    modelo = Sequential()
    modelo.add(Dense(units=num_entradas, activation="relu"))
    modelo.add(Dropout(dropout))   # con 0 no apaga ninguna neurona
    modelo.add(Dense(units=int(num_entradas / 2), activation="relu"))
    modelo.add(Dropout(dropout))
    modelo.add(Dense(units=1, activation="sigmoid"))
    modelo.compile(loss="binary_crossentropy", optimizer="adam")
    return modelo


def entrenar(X, y, dropout, pesos):
    tf.keras.utils.set_random_seed(SEMILLA)   # para obtener siempre los mismos resultados
    modelo = crear_modelo(X.shape[1], dropout)
    parar = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
    historial = modelo.fit(x=X, y=y, epochs=600, validation_split=0.2,
                           class_weight=pesos, callbacks=[parar], verbose=0)
    return modelo, historial


if __name__ == "__main__":
    X, y = cargar_train()
    # Configuración B (con Dropout 0.5), la elegida en el notebook 03
    modelo, historial = entrenar(X, y, dropout=0.5, pesos=calcular_pesos(y))
    modelo.save(RUTA_MODELO1)
    print("Modelo 1 guardado en", RUTA_MODELO1)