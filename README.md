# Proyecto B: Sistema de Predicción de Abandono de Clientes (Churn)

**Curso:** BD-151 Inteligencia Artificial Aplicada – Colegio Universitario de Cartago
**Profesor:** Osvaldo González Chaves
**Año:** 2026

## Integrantes

| Nombre | Carné | Correo |
|---|---|---|
| Gabriel Solano Molina | 120010053 | 120010053@cuc.cr |
| Jeremy Ramírez Pérez | 305650583 | 305650583@cuc.cr |
| Jeriel Fonseca Ramirez | 305720243 | 305720243@cuc.cr |
| Samiel Jimenez Portuguez | 305690316 | 305690316@cuc.cr |

## Descripción del problema

Sistema que predice qué clientes tienen mayor probabilidad de abandonar un servicio de telecomunicaciones. Identifica clientes en riesgo y los clasifica por nivel de urgencia para implementar estrategias de retención.

El sistema tiene dos redes neuronales, una API REST que las pone a disposición y un dashboard en Streamlit que consume la API.

## Dataset

- **Nombre:** Telco Customer Churn (Kaggle)
- **URL:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Registros:** 7 043 clientes
- **Variables:** 21 (demográficas, servicios contratados, información de cuenta)
- **Limpieza:** se eliminaron 11 registros con `TotalCharges` vacío (clientes con antigüedad 0). Quedan 7 032 clientes.
- **Desbalance:** cerca del 27 % de los clientes abandona.
- **División:** 80 % entrenamiento (5 625 clientes) y 20 % prueba (1 407 clientes), con `random_state=101`.
- **Variables de entrada:** 24, después de `pd.get_dummies` y de crear `num_servicios` (cantidad de servicios adicionales).

El archivo original está en `data/raw/` y no se modifica.

### Tasa de churn por segmento (EDA)

| Variable | Grupo de mayor riesgo | Grupo de menor riesgo |
|---|---|---|
| Contrato | Mensual: ≈ 43 % | Dos años: ≈ 3 % |
| Antigüedad | 0 a 12 meses: ≈ 48 % | 49 a 72 meses: ≈ 10 % |
| Método de pago | Cheque electrónico: ≈ 45 % | Tarjeta automática: ≈ 15 % |
| Internet | Fibra óptica: ≈ 42 % | Sin internet: ≈ 7 % |
| Servicios adicionales | 1 servicio: ≈ 46 % | 6 servicios: ≈ 5 % |

El análisis completo está en `notebooks/01_EDA.ipynb`.

## Modelos

| | Modelo 1: Clasificación binaria | Modelo 2: Scoring de riesgo |
|---|---|---|
| Pregunta | ¿El cliente se va o se queda? | ¿Qué tan probable es que se vaya? |
| Capas ocultas | 24 → 12 neuronas (relu), con `Dropout(0.5)` | 48 → 24 → 12 neuronas (relu), sin Dropout |
| Salida | 1 neurona sigmoid; "Se va" si la probabilidad es mayor a 0.5 | 1 neurona sigmoid: probabilidad de 0.0 a 1.0 y nivel de riesgo |
| Desbalance | `class_weight` = {0: 0.68, 1: 1.86} | Sin `class_weight`, para que la probabilidad sea real |
| EarlyStopping | `val_loss`, patience 25 | `val_loss`, patience 25, `restore_best_weights` |
| Archivo | `models/model1.keras` | `models/model2.keras` |

Lineamientos de entrenamiento comunes:

- Normalización con `MinMaxScaler` ajustado solo sobre el conjunto de entrenamiento (`models/scaler.pkl`).
- Variables categóricas con `pd.get_dummies`; la lista de columnas resultante se guarda en `models/columnas.pkl`.
- `loss='binary_crossentropy'`, `optimizer='adam'` y `validation_split=0.2`; el conjunto de prueba se usa solo para la evaluación final.
- Dos configuraciones por modelo (sin y con `Dropout`).

### Niveles de riesgo (Modelo 2)

| Nivel | Probabilidad de churn | Churn real en test | Acción sugerida |
|---|---|---|---|
| Bajo | menor a 0.10 | 2.8 % | Ninguna |
| Medio | de 0.10 a 0.30 | 20.0 % | Seguimiento sin costo |
| Alto | de 0.30 a 0.60 | 34.4 % | Oferta de retención |
| Crítico | 0.60 o más | 65.9 % | Contacto inmediato |

Los umbrales están en `models/umbrales_riesgo.json`, y la API y el dashboard los leen de ahí. La justificación está en `notebooks/04_ANN_Modelo2.ipynb`: el umbral de 0.30 coincide con el punto de equilibrio económico del análisis de ROI.

## API REST

Los dos endpoints reciben los mismos 19 datos del cliente (las columnas del dataset sin `customerID` ni `Churn`). Pydantic valida los datos: si falta un campo, un valor no es válido o la antigüedad no está entre 1 y 72 meses, la API responde con el error 422.

| Método | Endpoint | Modelo | Respuesta de ejemplo |
|---|---|---|---|
| POST | `/predict/churn` | Modelo 1 | `{"probabilidad": 0.8783, "prediccion": "Se va"}` |
| POST | `/predict/risk_score` | Modelo 2 | `{"probabilidad": 0.7862, "nivel": "Crítico"}` |

### Cómo usar la API

1. Encender la API desde la carpeta del proyecto:
```bash
   uvicorn api.main:app --reload
```
2. Abrir la documentación en el navegador: http://127.0.0.1:8000/docs
3. En cualquiera de los dos endpoints, presionar **Try it out**, pegar un cliente y presionar **Execute**.
4. Para apagar la API, presionar Ctrl + C en la terminal.

Cliente de riesgo alto (responde "Se va" y nivel "Crítico"):

```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
  "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
  "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
  "tenure": 5, "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 420.0
}
```

Cliente de riesgo bajo (responde "Se queda" y nivel "Bajo"):

```json
{
  "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
  "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
  "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
  "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
  "tenure": 60, "Contract": "Two year", "PaperlessBilling": "No",
  "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.0, "TotalCharges": 3900.0
}
```

Para ver la validación, cambiar `"Month-to-month"` por `"Weekly"` en el primer cliente: la API responde con el error 422.

## Dashboard

| Página | Qué hace |
|---|---|
| Home | Presenta el sistema, revisa que la API esté encendida, resume el dataset y muestra los niveles de riesgo. |
| Predicción | Se llenan los datos de un cliente y el dashboard consulta los dos endpoints: decisión del Modelo 1, probabilidad y nivel de riesgo del Modelo 2, y acción sugerida. |
| Análisis | Evalúa un grupo de clientes del dataset (de 20 a 300) por medio de la API, los segmenta por nivel de riesgo y permite filtrar y descargar la lista. |
| Métricas | Carga los dos modelos y los evalúa con el conjunto de prueba: comparación de métricas, matrices de confusión, churn real por nivel de riesgo y curvas de pérdida. |

## Estructura del proyecto

```
ProyectoAI-1/
│
├── README.md                      ← Guía de instalación y uso del proyecto
├── USO_DE_IA.md                   ← Uso de herramientas de IA de cada integrante
├── requirements.txt               ← Dependencias Python
├── .gitignore                     ← Archivos excluidos del control de versiones
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  ← Datos originales
│   └── processed/
│       ├── train.csv              ← Conjunto de entrenamiento
│       └── test.csv               ← Conjunto de prueba
│
├── notebooks/
│   ├── 01_EDA.ipynb               ← Análisis exploratorio de datos
│   ├── 02_Preprocesamiento.ipynb  ← Limpieza, variables dummy, normalización
│   ├── 03_ANN_Modelo1.ipynb       ← Entrenamiento del Modelo 1
│   ├── 04_ANN_Modelo2.ipynb       ← Entrenamiento del Modelo 2, niveles de riesgo y ROI
│   └── 05_Comparacion_Modelos.ipynb ← Comparación y selección de los modelos
│
├── src/
│   ├── __init__.py
│   ├── config.py                  ← Configuraciones globales (rutas, parámetros)
│   ├── data_prep.py               ← Funciones de preprocesamiento
│   └── train/
│       ├── __init__.py
│       ├── model1.py              ← Entrenamiento del Modelo 1
│       ├── model2.py              ← Entrenamiento del Modelo 2
│       └── utils.py               ← Utilidades compartidas (métricas, gráficas)
│
├── models/
│   ├── model1.keras               ← Modelo 1 guardado
│   ├── model2.keras               ← Modelo 2 guardado
│   ├── scaler.pkl                 ← MinMaxScaler entrenado
│   ├── columnas.pkl               ← Columnas finales tras get_dummies
│   ├── umbrales_riesgo.json       ← Umbrales de los niveles de riesgo
│   └── historial_modelo*_*.csv    ← Historiales de pérdida de las cuatro configuraciones
│
├── api/
│   ├── main.py                    ← Aplicación FastAPI con los endpoints
│   ├── schemas.py                 ← Modelos Pydantic para validación
│   └── predict.py                 ← Lógica de predicción
│
└── app/
    ├── Home.py                    ← Página principal del dashboard
    └── pages/
        ├── 1_Prediccion.py        ← Predicción individual
        ├── 2_Analisis.py          ← Análisis por lotes
        └── 3_Metricas.py          ← Métricas y rendimiento
```

## Instalación

Requisitos: Python 3.11 o superior (en una versión compatible con TensorFlow) y Git.

```bash
git clone https://github.com/jeriel169017-a11y/ProyectoAI-1.git
cd ProyectoAI-1
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

Los modelos ya vienen entrenados en `models/`, así que para usar el sistema no hace falta ejecutar los notebooks.

1. Encender la API desde la carpeta del proyecto y esperar el mensaje `Application startup complete`:
```bash
   uvicorn api.main:app --reload
```
2. En otra terminal, encender el dashboard. Se abre en http://localhost:8501:
```bash
   streamlit run app/Home.py
```

Los notebooks se pueden abrir con `jupyter notebook` para revisar el análisis. **Importante:** los notebooks 03 y 04 vuelven a entrenar los modelos y sobrescriben los archivos de `models/`. Los resultados pueden variar entre computadoras según la versión de TensorFlow y Keras, así que no se deben volver a ejecutar si se quieren conservar los resultados de este README.

## Entregables

- [x] Notebook de EDA con análisis de tasas de churn por segmento (`01_EDA.ipynb`)
- [x] Manejo de datos desbalanceados con `class_weight` (`03_ANN_Modelo1.ipynb` y la prueba en `04_ANN_Modelo2.ipynb`)
- [x] Dos modelos ANN: clasificación binaria y scoring de riesgo (`03` y `04`)
- [x] API REST con endpoints `/predict/churn` y `/predict/risk_score` (`api/`)
- [x] Dashboard Streamlit con segmentación de clientes por nivel de riesgo (`app/`)
- [x] Análisis de ROI potencial de estrategias de retención (`04_ANN_Modelo2.ipynb`)

## Resultados

### Configuraciones comparadas (validación)

| Configuración | Épocas entrenadas | Mejor época | Mejor val_loss | Elegida |
|---|---|---|---|---|
| Modelo 1 – A: sin Dropout | 37 | 12 | 0.4318 | |
| Modelo 1 – B: Dropout 0.5 | 83 | 58 | 0.4606 | Sí |
| Modelo 2 – A: sin Dropout | 30 | 5 | 0.4163 | Sí |
| Modelo 2 – B: Dropout 0.3 | 35 | 10 | 0.4193 | |

### Evaluación final con test (1 407 clientes, corte de 0.5)

| Modelo | Accuracy | Precisión | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Modelo 1 | 0.754 | 0.509 | 0.741 | 0.603 | 0.839 |
| Modelo 2 | 0.796 | 0.593 | 0.608 | 0.601 | 0.840 |

Precisión, recall y F1 corresponden a la clase "se va". De los 355 clientes de test que se fueron, el Modelo 1 detectó 263 con 254 falsas alarmas, y el Modelo 2 detectó 216 con 148 falsas alarmas.

**Modelo seleccionado y justificación:**

Ningún modelo es mejor en todo: los dos tienen casi el mismo AUC, es decir, ordenan igual de bien a los clientes. Por eso cada uno se usa para lo que hace mejor:

- **Modelo 1 (configuración B, con Dropout 0.5):** se eligió sobre la A, aunque esta tenía menor `val_loss`, porque detecta más clientes que se van (recall de 0.75 contra 0.65 en validación, y F1 de 0.62 contra 0.60). Para la empresa es más costoso perder a un cliente que contactar a uno de más. En test mantuvo un recall de 0.741. Responde `/predict/churn`.
- **Modelo 2 (configuración A, sin Dropout):** tuvo la menor `val_loss`, y la elección se confirmó repitiendo el entrenamiento con 5 semillas. Su probabilidad media en test es 28.5 %, cercana al churn real de 25.2 %, mientras que el Modelo 1 da 38.8 % por el efecto de `class_weight`. Por eso es el que define los niveles de riesgo. Responde `/predict/risk_score`.

La comparación completa está en `notebooks/05_Comparacion_Modelos.ipynb`.

### Análisis de ROI

Supuestos (notebook 04): contactar a un cliente cuesta $5 y la oferta de retención regala 1 mes de servicio; el 30 % de los clientes en fuga que reciben la oferta se queda; un cliente retenido aporta 12 meses de ingreso. Los montos son ingresos sobre los 1 407 clientes de test.

| Estrategia | Clientes contactados | Fugas atendidas | Resultado neto | ROI |
|---|---|---|---|---|
| Solo Crítico | 255 | 168 | $27 245 | 121 % |
| Crítico + Alto | 560 | 273 | $29 903 | 65 % |
| Crítico + Alto + Medio | 900 | 341 | $23 682 | 35 % |
| Todos (sin modelo) | 1 407 | 355 | −$2 802 | −3 % |

## Conclusiones y recomendaciones

**Conclusiones**

- El cliente con mayor riesgo de abandonar es nuevo, tiene contrato mensual y fibra óptica, paga con cheque electrónico y no tiene seguridad en línea ni soporte técnico.
- Los dos modelos ordenan igual de bien a los clientes (AUC ≈ 0.84). La diferencia está en el uso: el Modelo 1 detecta el 74 % de las fugas a cambio de más falsas alarmas, y el Modelo 2 da probabilidades cercanas a la realidad.
- Los niveles de riesgo separan bien a los clientes: el churn real pasa de ≈ 3 % en el nivel Bajo a ≈ 66 % en el Crítico.
- Usar el modelo para decidir a quién ofrecer retención convierte una pérdida (ofrecer a todos: −$2 802) en un resultado positivo de ≈ $30 000.

**Recomendaciones**

- Ofrecer retención a los niveles Crítico y Alto; con presupuesto limitado, empezar por Crítico. Al nivel Medio, solo seguimiento sin costo, y al Bajo, ninguna acción.
- Ofrecer beneficios para pasar a los clientes de contrato mensual a contratos de uno o dos años.
- Dar seguimiento especial durante el primer año, que es cuando más clientes se van.
- Promover los servicios de seguridad en línea y soporte técnico, porque los clientes que los tienen se quedan más.
- Medir la tasa de éxito real de la oferta, porque el ROI depende de ese supuesto: con un 25 % de éxito, contactar al nivel Alto por sí solo ya daría pérdida.

## Uso de IA

El uso de herramientas de IA de cada integrante está documentado en [`USO_DE_IA.md`](USO_DE_IA.md).