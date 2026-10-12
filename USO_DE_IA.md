# Uso de IA en el proyecto

## Jeriel

### En qué usamos la IA

Yo usé Claude para que me explicara ciertas partes que debía hacer ya que me tocó trabajar con lo que mis compañeros iban haciendo, pero cierta parte del código no entendía y preferí saber qué significaban ciertas palabras como get_dummies o MinMaxScaler que no me quedaban claros los conceptos del todo. Luego para traducir variables al español, sinceramente no sabía qué significaba "Tenure", al ver el significado ya logré entender el motivo de que 11 registros estaban en blanco y no con un número. Para la limpieza quise usar material de cursos pasados pero al ser en R debía cambiar ciertas partes para usarlo acá. Al hacer el preprocesado quería que estuviera muy completo y le pedí sugerencias para ver si funcionaban a nuestro proyecto.

### Prompts

1. "Estoy haciendo un EDA, pero me gustaría saber qué observaciones o sugerencias me das para hacerlo más completo (le envié una captura de lo que tenía)"

2. "Debo hacer una comparación de modelos para una red neuronal, pero no me queda claro qué debo agregar de cada modelo para compararlos"

3. "Este proyecto es sobre redes neuronales, el profesor en clase nos habló sobre que no todas las variables se iban a ocupar, estoy con el EDA ¿en este paso ya debo saber cuáles variables quitar o eso se ve más adelante?"

### Reflexión

En lo personal hacer este tipo de proyectos mientras que me voy apoyando de la IA me ha servido mucho, puedo decir que aprendo bastante incluso más que en algunos cursos, sinceramente la programación no me gustaba tanto porque no he tenido buenos cursos sobre eso, hasta que llegué a PrograII y ya el profe sí sabía explicar, nos ponía a programar de 0, pero por no tener buenas bases a veces sí me llega a costar algo, pero cada proyecto voy aprendiendo nuevas cosas, en lo personal me gusta más hacer proyectos que exámenes siento que aprendo más.

Por ejemplo, en este proyecto aprendí que puedo clonar un repositorio de git desde PyCharm, crear carpetas, notebooks desde la terminal, no solo dándole clic derecho y darle new y todo eso, al inicio quería hacer cosas pero debía instalar los requerimientos, librerías y eso, pero me di cuenta que desde la consola puedo instalar todo eso sin tener que ir uno a uno, sí dura unos 2 minutos pero siento que vale la pena. En este proyecto no hice la parte de modelos, me incliné por limpieza y EDA, entonces sí me gustaría aprender más, pero viendo el trabajo que hicieron mis compañeros con cada modelo me llevo algo.

### Modificaciones/Análisis generado

Sí modifiqué ciertas partes, ya que, al momento de ejecutar ciertas casillas me daban error y no sabía el porqué o no recordaba cómo unir dos variables en una. Sobre cosas generadas del todo, algunas visualizaciones que recordaba que explicaban bien y hacían más fácil la interpretación de los datos pero no recordaba el cómo se hacían.



## Jeremy

### En qué usamos la IA

**Claude (Anthropic):** la usé para entender temas que no venían en las diapositivas y que se usaron en el proyecto.

| Tema | Para qué se usó en el proyecto |
| --- | --- |
| validation_split | Separar la validación del entrenamiento para usar el test solo en la evaluación final. En el caso práctico de clase se validaba con validation_data usando el test. |
| class_weight y compute_class_weight | Calcular pesos para el desbalance de clases (27 % de churn). En el Modelo 2 se probó y se descartó. |
| Efecto de class_weight en las probabilidades | Explicar por qué el Modelo 2 no lo usa: inflaba la probabilidad de churn (42 % contra 26 % real). |
| restore_best_weights en EarlyStopping | Quedarse con los pesos de la mejor época y no con los de la última. |
| Comparación con varias semillas (set_random_seed) | Comprobar que la diferencia entre la versión con y sin Dropout no era suerte. |
| AUC y curva ROC | Medir qué tan bien el modelo ordena a los clientes de mayor a menor riesgo. |
| Calibración (probabilidad predicha contra churn real) | Comprobar que la probabilidad del modelo es real. |
| Niveles de riesgo a partir de la probabilidad | Convertir la probabilidad en Bajo, Medio, Alto y Crítico con umbrales justificados. |
| inverse_transform del MinMaxScaler | Recuperar MonthlyCharges en dólares para el análisis de ROI. |
| ROI y punto de equilibrio | Elegir a qué clientes ofrecer retención y justificar el umbral de 0.30. |
| Análisis de sensibilidad | Ver desde qué tasa de éxito de la oferta es rentable cada estrategia. |
| pd.cut | Agrupar las probabilidades por rangos para revisar la calibración y definir los umbrales. |

### Prompts

1. "Necesito que me ayudes a poder entregar un proyecto mas profesional, ya tienes el contexto del proyecto, necesito que lo revises y me des ideas de que funciones puede agregarle aunque no estén en los pdfs de clase, y que me ayudes a implemnetarlos"

2. "Hagamos una discusión de porque debería implementar dichas funciones a mi proyecto y en que cosas me ayudan o si mas bien están de mas en mi proyecto y no aportan nada importante o relevante"

3. "Puedes hacer una revisión profunda de el proyecto ya terminado para que analices si todo esta correcto, si hay cosas que corregir, problemas críticos etc"

### Reflexión

La IA me ayudó a implementar cosas nuevas no vistas en clase al igual que revisar cosas que ya habíamos visto en clase,  por ejemplo al preguntar por qué mi curva de validación subía, entendí qué es el sobreajuste y cómo EarlyStopping se queda con la mejor época. También aprendí temas que no vimos en clase, como el AUC, por qué class_weight infla las probabilidades y cómo calcular un punto de equilibrio para el ROI.

También aprendí que la IA puede equivocarse. La primera versión tenía un error de fondo que solo apareció cuando pedí una revisión estricta. Eso me enseñó a verificar los resultados y a entender cada decisión.

### Modificaciones/Análisis generado

- Después de pedir una revisión estricta, se corrigió la primera versión. Se quitó class_weight, se agregó la prueba con 5 semillas y los umbrales pasaron de 0.2 / 0.4 / 0.8 a 0.10 / 0.30 / 0.60.

- Verificación propia: ejecuté el notebook en PyCharm y revisé los resultados y los gráficos.



## Gabriel

### En qué usamos la IA

**Claude (Anthropic):** lo usé como apoyo en mi parte del proyecto: los notebooks 01 a 03 y el dashboard. Antes de empezar le di como referencia el material del curso, lo usé para:

- **Identificar posibles correcciones:** revisó los notebooks comparándolos con lo visto en clase y me señaló qué cambiar; por ejemplo, limpiar `TotalCharges` con `replace` y `dropna`, cambiar `pd.cut` por una función con `apply` y dividir los datos con `random_state=101`.
- **Proponer y explicar el código:** la herramienta me ayudo a identificar el verdadero valor de los gráficos y resultados obtenidos a lo largo del proyecto, esto con el fin de poder comprender bien que estabamos trabajando.
- **Resolver errores** que me salieron al ejecutar.
- **Revisar lo que subían mis compañeros** (la API, el notebook 04 y el notebook de comparación) antes de seguir con mi parte, esto ya que Claude cuenta con la función de poder analizar y recomendar funciones dentro del repositorio de github.

| Tema | Para qué se usó en el proyecto |
| --- | --- |
| class_weight | Darle más peso a los clientes que se van (27 % del train) para que el Modelo 1 no los deje escapar. Los pesos se calcularon con total / (2 × cantidad de la clase). |
| validation_split | Separar el último 20 % de train para validar sin usar el test. |
| tf.keras.utils.set_random_seed | Que al volver a ejecutar el notebook 03 salgan los mismos resultados. |
| Streamlit | Construir el dashboard de cuatro páginas. |
| requests | Enviar los datos de los clientes desde el dashboard a la API. |
| st.session_state y st.cache_data | Que la página de análisis no pierda los resultados y que la de métricas no recargue los modelos cada vez. |
| classification_report con output_dict y roc_auc_score | Sacar cada métrica por separado para la tabla comparativa y calcular el AUC. |
| pathlib.Path | Encontrar los archivos del proyecto desde cualquier página del dashboard. |

### Prompts

1. "Te adjunto el material de teoría y ejemplos brindados en el curso, para que puedas saber realmente de lo que estamos trabajando y no agregues material extra al proyecto."

2. "Quisiera que analices estos documentos y me recomiendes que mejoras se le pueden hacer, ten en cuenta el material enviado anteriormente para poder hacer alguna corrección."

3. "No entiendo mucho esta parte del código, que valor da al archivo? es realmente necesario trabajarlo asi o crees que puedo cambiarlo para lograr un mejor resultado?"

### Reflexión

Al realizar este proyecto me pude dar cuenta como la IA mas alla de ser una nueva "herramienta", ya se esta volviendo parte de nuestra vida cotidiana, no solo nosotros como estudiantes de Big Data la utilizamos para apoyarnos en nuestros proyectos y en nuestro estudio, ahora ya mas campos la utilizan como una de las piezas mas importantes de su estructura.
Siento que, al aplicarla en este trabajo, nos demostró hasta donde realmente puede llegar la IA, mas alla de solo pedirle cosas, nos mostro a como realmente pedirle ayuda, nutrirla aun mas de conocimiento, y sobretodo, lograr llegar a un aprendizaje en conjunto, no solo simplemente un "copiar y pegar".

### Modificaciones/Análisis generado

- No descargué archivos terminados: escribí yo el código en los archivos y ejecuté cada celda y cada página en mi computadora para comprobar los resultados.
- Corregí los errores que salieron al escribirlo: en el notebook 03 la celda 26 daba `NameError` porque faltaba la celda `model = model_b`, y en `1_Prediccion.py` faltaban los dos puntos de un `if`.
- Adapté los comentarios del dashboard al estilo del profesor (`# ... #`) y cambié algunos textos de `Home.py`.
- Al ejecutar el notebook 04 en mi computadora los resultados cambiaron, porque tengo otra versión de Keras. No subí esos resultados: restauré la carpeta `models` y el notebook se volvió a ejecutar con Claude, para conservar los resultados originales de Jeremy.