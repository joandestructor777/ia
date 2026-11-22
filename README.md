# Pulso · Tamizaje Metabólico & Predictor Clínico con Naïve Bayes
> **Actividad 3 — Inteligencia Artificial I**  
> **Integrantes (Grupo 3):** Joan Camilo González Camacho & Juan David Ruiz Carvajalino  
> **Repositorio Oficial:** [github.com/joandestructor777/ia](https://github.com/joandestructor777/ia)  
> **Stack Tecnológico:** Python 3.11 · Scikit-Learn · Streamlit · Pandas · NumPy · Joblib · Pillow · WinOCR

---

## Índice General

1. [De Dónde Venimos y de Qué Trata este Proyecto](#1-de-dónde-venimos-y-de-qué-trata-este-proyecto)
2. [El Problema Clínico y los Datos Reales](#2-el-problema-clínico-y-los-datos-reales)
3. [El Choque con la Realidad: Detección y Limpieza de Ceros Biológicos](#3-el-choque-con-la-realidad-detección-y-limpieza-de-ceros-biológicos)
4. [Entendiendo el Algoritmo: ¿Por Qué Naïve Bayes y por qué GaussianNB?](#4-entendiendo-el-algoritmo-por-qué-naïve-bayes-y-por-qué-gaussiannb)
5. [Preparación del Entorno, Dependencias y Librerías](#5-preparación-del-entorno-dependencias-y-librerías)
6. [El Paso a Paso del Código y Entrenamiento (`01_training.ipynb`)](#6-el-paso-a-paso-del-código-y-entrenamiento-01_trainingipynb)
7. [Evaluación Honesta de los Resultados: Métricas en Contexto Médico](#7-evaluación-honesta-de-los-resultados-métricas-en-contexto-médico)
8. [Evolución de la Aplicación Web: De Prototipo Rústico a HealthTech "Pulso"](#8-evolución-de-la-aplicación-web-de-prototipo-rústico-a-healthtech-pulso)
9. [Errores que Nos Salieron en el Proceso y Cómo los Resolvimos](#9-errores-que-nos-salieron-en-el-proceso-y-cómo-los-resolvimos)
10. [Despliegue Continuo en la Nube (Streamlit Cloud + GitHub)](#10-despliegue-continuo-en-la-nube-streamlit-cloud--github)
11. [Estructura del Repositorio](#11-estructura-del-repositorio)
12. [Cómo Ejecutar el Proyecto en tu Máquina Local](#12-cómo-ejecutar-el-proyecto-en-tu-máquina-local)
13. [Reflexiones Finales y Aprendizajes de la Actividad](#13-reflexiones-finales-y-aprendizajes-de-la-actividad)

---

## 1. De Dónde Venimos y de Qué Trata este Proyecto

En la **Actividad 2** de la materia veníamos trabajando con **PCA (Análisis de Componentes Principales)**, explorando cómo reducir la dimensionalidad de conjuntos de datos complejos para visualizarlos en planos de dos o tres dimensiones sin perder tanta variabilidad. Fue un ejercicio interesante de álgebra lineal aplicada, pero para esta **Actividad 3** el profesor nos reasignó un reto de aprendizaje supervisado: **el algoritmo Naïve Bayes**, con el objetivo puntual de entrenar un modelo y **desplegarlo como una aplicación funcional en la nube**.

Al comenzar, como estudiantes nos hicimos una pregunta clave: *¿Nos limitamos a hacer un notebook que cumpla los requisitos mínimos con un formulario feo, o intentamos entender cómo se construye un producto de Machine Learning de punta a punta (MLOps básico)?*

Optamos por lo segundo. Decidimos construir **Pulso**, una plataforma de tamizaje temprano de riesgo de diabetes mellitus que no solo carga un modelo serializado, sino que incluye ingestión óptica de exámenes médicos reales de laboratorio (OCR con cámara y archivo), interfaz gráfica sobria inspirada en aplicaciones de salud modernas (*Apple Health / One Medical*) y orientaciones clínicas preventivas basadas en las guías del **Ministerio de Salud y Protección Social de Colombia**.

---

## 2. El Problema Clínico y los Datos Reales

### 2.1. ¿Por qué Diabetes Mellitus?
Elegimos la predicción de diabetes porque es un problema de salud pública de enorme impacto en Colombia y en toda Latinoamérica. Millones de personas cursan con estados de resistencia a la insulina o prediabetes sin saberlo. Además, es un caso de uso donde las variables son numéricas, directas y fáciles de contrastar contra un reporte de laboratorio rutinario.

### 2.2. El Conjunto de Datos: Pima Indians Diabetes
Trabajamos con el conocido dataset **Pima Indians Diabetes** (alojado en UCI Machine Learning Repository y OpenML), el cual contiene **768 registros reales** de pacientes mujeres mayores de 21 años evaluadas en centros de salud. Supera ampliamente el mínimo de 500 filas requerido por la guía de la actividad.

El dataset cuenta con 8 características biomédicas explicativas y 1 variable objetivo binaria:

| Característica | Nombre en Dataset | Tipo de Variable | Significado Fisiológico |
|---|---|---|---|
| Embarazos | `Pregnancies` | Entero discreto | Número de gestaciones previas de la paciente |
| Glucemia | `Glucose` | Numérico continuo (mg/dL) | Concentración de glucosa en plasma a las 2 horas de una prueba de tolerancia oral |
| Presión Arterial | `BloodPressure` | Numérico continuo (mm Hg) | Presión arterial diastólica |
| Pliegue Cutáneo | `SkinThickness` | Numérico continuo (mm) | Grosor del pliegue cutáneo del tríceps (reserva grasa) |
| Insulina | `Insulin` | Numérico continuo (µU/mL) | Nivel de insulina sérica a las 2 horas |
| Índice de Masa Corporal | `BMI` | Numérico continuo (kg/m²) | Peso en kg dividido por altura al cuadrado ($kg/m^2$) |
| Función de Pedigrí | `DiabetesPedigreeFunction` | Numérico continuo (índice) | Puntuación genética calculada según antecedentes familiares de diabetes |
| Edad | `Age` | Entero discreto (años) | Edad de la paciente |
| **Diagnóstico** | `Outcome` | **Binario (0 ó 1)** | **0: No diabética / Sano \| 1: Diagnóstico positivo de diabetes** |

---

## 3. El Choque con la Realidad: Detección y Limpieza de Ceros Biológicos

Cuando abrimos por primera vez el archivo CSV en el notebook y ejecutamos `df.describe()`, nos llevamos una sorpresa que fue nuestro primer gran aprendizaje de datos:

```python
# Conteo de ceros en variables donde fisiológicamente es imposible tener cero
columnas_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
(df[columnas_ceros] == 0).sum()
```

Nos encontramos con cantidades absurdas de ceros:
- **Glucosa con valor 0:** 5 pacientes.
- **Presión arterial con valor 0:** 35 pacientes.
- **Grosor de piel con valor 0:** 227 pacientes.
- **Insulina con valor 0:** 374 pacientes (¡casi la mitad del dataset!).
- **Índice de masa corporal (BMI) con valor 0:** 11 pacientes.

### ¿Qué significaba esto en el mundo real?
Una persona viva **no puede tener 0 de glucosa en sangre** ni **0 de presión arterial diastólica**; eso equivaldría a muerte biológica. Lo que ocurrió es que los analistas que tabularon el dataset hace años usaron el número `0` como marcador de **"dato no tomado"** o valor nulo (*missing value*).

### El dilema del novato: ¿Borrar filas o Imputar?
Al principio pensamos: *"Bueno, hagamos `df.dropna()` y quitemos todas las filas con ceros"*.  
Pero al probarlo nos dimos cuenta de que si eliminábamos todas las filas donde la insulina o la piel tenían 0, **el dataset pasaba de 768 filas a menos de 392**. Estábamos destruyendo la mitad de la información y sesgando la muestra.

### ¿Por qué imputamos con la Mediana y no con la Media?
Decidimos reemplazar los ceros por valores estadísticos de reemplazo (imputación). Pero no usamos la media aritmética (`mean`), sino la **mediana** (`median`).  
¿Por qué? Porque al graficar los histogramas de variables como la insulina o el grosor de piel, notamos que tienen una **asimetría positiva severa** (muchos valores concentrados en números bajos y unos pocos pacientes con picos extremos de 400 u 800 de insulina). La media se deja arrastrar fácilmente por esos valores atípicos (*outliers*), inflando artificialmente el valor imputado. La mediana, en cambio, representa el percentil 50 real y es inmune a valores extremos.

Implementamos esta limpieza modular en `src/preprocessing.py`:

```python
import pandas as pd
import numpy as np

columnas_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def limpiar_datos(df):
    df_limpio = df.copy()
    for col in columnas_ceros:
        if col in df_limpio.columns:
            # Marcamos los ceros como NaN y los rellenamos con la mediana
            df_limpio[col] = df_limpio[col].replace(0, np.nan)
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
    return df_limpio
```

Con esto logramos un dataset 100% sano y coherente que guardamos en `data/processed/dataset_clean.csv`.

---

## 4. Entendiendo el Algoritmo: ¿Por Qué Naïve Bayes y por qué GaussianNB?

Al estudiar la teoría para la sustentación, queríamos entender con claridad qué hace el algoritmo tras bambalinas.

### 4.1. El Fundamento: Teorema de Bayes
Naïve Bayes se basa en la conocida ecuación de probabilidad condicional:

$$P(C_k \mid X) = \frac{P(X \mid C_k) \cdot P(C_k)}{P(X)}$$

Donde:
- $C_k$ es la clase que queremos predecir ($C_0 = \text{Sano}$, $C_1 = \text{Diabetes}$).
- $X = (x_1, x_2, \dots, x_8)$ son las 8 variables del paciente (glucosa, presión, edad, etc.).
- $P(C_k)$ es la **probabilidad a priori**: qué tan común es la diabetes en la población del dataset (alrededor del 35%).
- $P(X \mid C_k)$ es la **verosimilitud** (*likelihood*): qué tan probable es observar esos niveles de glucosa, insulina o presión en personas que ya sabemos que son sanas o enfermas.
- $P(C_k \mid X)$ es la **probabilidad a posteriori**: la probabilidad calculada de que el paciente pertenezca a la clase $C_k$ dados sus análisis clínicos.

### 4.2. ¿Por qué se llama "Naïve" (Ingenuo)?
El algoritmo hace una asunción muy fuerte: **asume que todas las características $x_i$ son condicionalmente independientes entre sí dado el diagnóstico**.  
En medicina sabemos que esto en la práctica no es 100% cierto: una persona con IMC alto suele tener una glucemia e insulina más elevadas. Sin embargo, matemáticamente asumir independencia permite multiplicar las probabilidades individuales:

$$P(X \mid C_k) = \prod_{i=1}^{n} P(x_i \mid C_k)$$

Esta simplificación reduce drásticamente el costo computacional, evita que el modelo requiera cientos de miles de parámetros y lo hace increíblemente rápido y resistente frente a muestras moderadas.

### 4.3. ¿Por qué elegimos Gaussian Naïve Bayes (`GaussianNB`)?
En Scikit-Learn existen varias variantes de Naïve Bayes:
1. `MultinomialNB`: Diseñado para datos discretos y recuentos de palabras (el típico filtro de SPAM).
2. `BernoulliNB`: Diseñado para características booleanas (si una palabra está o no está presente, 0 ó 1).
3. `GaussianNB`: **Diseñado para datos numéricos continuos**.

Dado que la glucosa (ej: 120.5 mg/dL), el IMC (ej: 28.4 kg/m²) y la presión (ej: 76.0 mm Hg) son medidas biológicas continuas, **GaussianNB** es la opción teórica y práctica correcta. Lo que hace el algoritmo es calcular la media ($\mu$) y la varianza ($\sigma^2$) de cada variable para cada clase, y modelar la probabilidad con la campana de Gauss:

$$P(x_i \mid C_k) = \frac{1}{\sqrt{2\pi\sigma_{ik}^2}} \exp\left( -\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2} \right)$$

---

## 5. Preparación del Entorno, Dependencias y Librerías

Para que cualquier persona pueda replicar el proyecto en su máquina o en servidores cloud, organizamos las librerías con mucho cuidado.

### 5.1. ¿Por qué usar Entorno Virtual (`venv`)?
Al principio instalábamos cosas en el Python global de nuestro equipo, pero nos dimos cuenta de que chocaban versiones de otras materias. Aprendimos la buena práctica de aislar siempre las dependencias:

```bash
# 1. Crear el entorno virtual en la carpeta del proyecto
python -m venv venv

# 2. Activar el entorno virtual
# En Windows (PowerShell):
venv\Scripts\Activate.ps1
# En Windows (CMD):
venv\Scripts\activate.bat
# En Linux / Mac:
source venv/bin/activate

# 3. Actualizar el gestor de paquetes pip
python -m pip install --upgrade pip

# 4. Instalar todas las librerías fijadas
pip install -r requirements.txt
```

### 5.2. El Archivo `requirements.txt` y para qué sirve cada librería
Nuestro `requirements.txt` contiene exactamente lo que el proyecto necesita:

```text
streamlit
scikit-learn
pandas
numpy
joblib
matplotlib
seaborn
Pillow
winocr
```

- **`streamlit`:** Framework web para levantar la aplicación interactiva en Python puro sin tener que programar HTML/CSS/JS desde cero.
- **`scikit-learn`:** La librería madre del Machine Learning en Python; nos proporciona el modelo `GaussianNB`, la separación de muestras y las funciones de evaluación.
- **`pandas` y `numpy`:** Manejo eficiente de dataframes, matrices numéricas y preprocesamiento de datos.
- **`joblib`:** Herramienta de serialización optimizada para guardar y cargar objetos pesados de Scikit-Learn en formato binario `.pkl`.
- **`matplotlib` y `seaborn`:** Generación de gráficos en el análisis exploratorio (matriz de correlación y matriz de confusión).
- **`Pillow` (PIL):** Procesamiento, lectura y conversión de imágenes cargadas por el usuario o capturadas por la cámara web.
- **`winocr`:** Motor de Reconocimiento Óptico de Caracteres nativo de Windows (Windows Media OCR), ligero y sin dependencias externas pesadas como Tesseract.

---

## 6. El Paso a Paso del Código y Entrenamiento (`01_training.ipynb`)

Todo el flujo experimental quedó registrado de forma transparente en el notebook interactivo ubicado en `notebooks/01_training.ipynb`. El proceso se dividió en 6 etapas:

### Paso 1: Exploración y Análisis de Correlación
Inspeccionamos la correlación de Pearson entre las variables para entender cuáles pesaban más en la aparición de diabetes:
```python
plt.figure(figsize=(8, 6))
sns.heatmap(df_limpio.corr(), annot=True, cmap='Blues', fmt='.2f')
```
Descubrimos que la **Glucosa** ($r \approx 0.49$) y el **IMC** ($r \approx 0.31$) son las variables con mayor correlación lineal positiva directa frente al diagnóstico.

### Paso 2: Partición Estratificada Train / Test
Para no autoengañarnos evaluando el modelo con los mismos datos que vio al entrenar, dividimos la muestra en un **80% de entrenamiento (614 pacientes)** y un **20% de prueba (154 pacientes)**:

```python
from sklearn.model_selection import train_test_split

X = df_limpio.drop(columns=['Outcome'])
y = df_limpio['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)
```
> **Nota de aprendizaje:** Usamos `stratify=y` para obligar a que tanto el conjunto de entrenamiento como el de prueba mantuvieran exactamente el mismo porcentaje de pacientes sanos (65.1%) y con diabetes (34.9%), evitando sesgos de partición.

### Paso 3: Ajuste del Modelo GaussianNB
El entrenamiento con Naïve Bayes es instantáneo porque no itera cientos de épocas con gradiente descendente, sino que calcula sumas y desviaciones estándar analíticas en milisegundos:

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

### Paso 4: Evaluación en Datos Nuevos
Generamos las predicciones en `X_test` y evaluamos métricas reales:

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### Paso 5: Serialización con Joblib
Para no tener que volver a entrenar el modelo cada vez que un usuario abre la web, exportamos el objeto ajustado a un archivo binario `.pkl`:

```python
import joblib
joblib.dump(model, '../models/modelo.pkl')
```

---

## 7. Evaluación Honesta de los Resultados: Métricas en Contexto Médico

Al correr la evaluación en los 154 pacientes de prueba, obtuvimos los siguientes resultados:

```text
Accuracy global: 70.13%

Reporte de Clasificación:
              precision    recall  f1-score   support
           0       0.79      0.74      0.76       100
           1       0.57      0.63      0.60        54

    accuracy                           0.70       154
   macro avg       0.68      0.68      0.68       154
weighted avg       0.71      0.70      0.70       154
```

### ¿Qué nos dicen estos números como futuros ingenieros?
1. **La Exactitud (~70%):** El modelo clasifica de forma correcta a 7 de cada 10 pacientes en un conjunto ciego que jamás vio durante el entrenamiento. Para ser un modelo probabilístico simple con sólo 8 variables clínicas, es un punto de partida muy decente.
2. **La Diferencia entre Precisión y Recall en Medicina:**
   - En la clase 1 (personas con diabetes), la **precisión es 0.57** y el **recall (sensibilidad) es 0.63**.
   - Esto significa que de cada 100 personas que realmente tienen diabetes en la prueba, el modelo logra alertar oportunamente a 63.
   - En medicina preventiva, **un Falso Negativo es mucho más peligroso que un Falso Positivo**: decirle a una persona enferma que está sana puede impedir que busque tratamiento a tiempo; en cambio, alertar a una persona sana de que su riesgo es moderado simplemente la motiva a realizarse una prueba de sangre de control y mejorar sus hábitos.
3. **Por qué Naïve Bayes tiene limitaciones aquí:**
   Dado que asume independencia total entre variables, cuando la glucosa y la insulina se elevan juntas, el modelo a veces "sobreestima" o "subestima" la probabilidad combinada. Entender esta limitación matemática nos sirvió para saber cómo presentar los resultados en la aplicación web: **como un porcentaje de probabilidad de tamizaje y orientación preventiva, jamás como una sentencia médica definitiva**.

---

## 8. Evolución de la Aplicación Web: De Prototipo Rústico a HealthTech "Pulso"

Uno de los aspectos de los que más aprendimos en este proyecto fue cómo diseñar una interfaz que sirva de verdad a usuarios comunes.

```
Versión 1 (Básica) ─────────► Versión 2 ("Pulso · Tamizaje Metabólico")
• Formulario plano            • Navbar Edge-to-Edge estilo Apple Health / One Medical
• Colores semáforo estridentes • Paleta sobria (Ámbar suave y Teal de estabilidad)
• Entrada solo manual         • Ingestión Documental: Cámara Web + Carga de Archivos + OCR
• Diagnóstico frío            • Orientación clínica alineada con MinSalud Colombia
```

### 8.1. La Versión Inicial (`app_original_backup.py`)
Nuestra primera versión de Streamlit era el clásico formulario con `st.number_input` y un botón que tiraba `st.error('Positivo para Diabetes')` en rojo chillón o `st.success('Negativo')` en verde.  
Funcionaba, pero al probarla sentimos que:
- Parecía un ejercicio escolar apresurado.
- El color rojo sangre en una app de salud genera pánico y estrés en el usuario.
- Obligaba al usuario a transcribir a mano 8 números que normalmente vienen impresos en una hoja de laboratorio.

### 8.2. El Rediseño: Pulso · Salud Digital (`app/app.py`)
Decidimos rehacer la aplicación desde sus cimientos:

1. **Diseño Visual HealthTech:** Inyectamos estilos CSS personalizados en Streamlit para simular plataformas modernas como *One Medical*, *Linear* o *Apple Health*:
   - Tipografía limpia con escala tipográfica calculada (`Inter`, `-apple-system`).
   - Barra de navegación fija superior (*Edge-to-edge navbar*) con estado de protocolo y versión del modelo.
   - Tarjetas blancas flotantes con bordes sutiles (`surface-card`).
2. **Psicología del Color Clínico:**
   - Eliminamos por completo los rojos y verdes estridentes.
   - Si el riesgo estimado es elevado ($\ge 50\%$), usamos un **ámbar cálido** (`#D97706`) con el distintivo *"Atención Preventiva Sugerida"*.
   - Si el riesgo es bajo, usamos un **verde azulado suave / teal** (`#0D9488`) con el distintivo *"Rango Fisiológico Favorable"*.
3. **Ingestión Documental con OCR (Cámara y Archivo):**
   - Agregamos un selector para capturar el examen de laboratorio directamente con la **cámara del computador/celular** (`st.camera_input`), **cargar una imagen JPG/PNG** o probar con un **examen de muestra colombiano** (`ejemplo_examen_colombia.png`).
   - Creamos una rutina de extracción que procesa la imagen, ordena espacialmente las líneas del reporte y busca con expresiones regulares biomarcadores como *Glucemia en ayunas*, *Insulina basal*, *Tensión arterial*, *IMC* y *Gestaciones*.
   - Los datos detectados se muestran en pequeños *chips interactivos* y rellenan automáticamente el formulario para que el usuario solo tenga que verificar y ajustar si hace falta.
4. **Pautas de Cuidado según el Ministerio de Salud de Colombia:**
   - En lugar de decir simplemente "tienes diabetes", el resultado entrega la **probabilidad posterior calculada por Naïve Bayes** y un bloque de recomendaciones preventivas: sugerencia de prueba de *Hemoglobina Glicosilada (HbA1c)*, consulta preventiva con médico general en su EPS/IPS y pautas de 150 minutos semanales de actividad física.

---

## 9. Errores que Nos Salieron en el Proceso y Cómo los Resolvimos

Durante el desarrollo nos topamos con varios fallos reales de programación y entorno. Documentar cómo los solucionamos fue uno de los mayores aprendizajes:

### Error 1: `FileNotFoundError` al cargar el modelo serializado en Streamlit
- **Qué pasaba:** Cuando corríamos `streamlit run app/app.py` desde la raíz del repositorio, `joblib.load('../models/modelo.pkl')` funcionaba bien. Pero si abríamos la terminal parados dentro de la carpeta `app/`, o si Streamlit Cloud arrancaba el proceso desde otra ruta de ejecución, el programa se caía con:
  ```text
  FileNotFoundError: [Errno 2] No such file or directory: '../models/modelo.pkl'
  ```
- **Cómo lo resolvimos:** Aprendimos a no confiar en rutas relativas al directorio de trabajo actual (`cwd`), sino a construir rutas ancladas a la ubicación real del archivo de código con `__file__`:
  ```python
  ruta_modelo = os.path.join(os.path.dirname(__file__), '..', 'models', 'modelo.pkl')
  if not os.path.exists(ruta_modelo):
      ruta_modelo = os.path.join(os.path.dirname(__file__), 'models', 'modelo.pkl')
  model = joblib.load(ruta_modelo)
  ```

### Error 2: Advertencias de Scikit-Learn por nombres de características ausentes
- **Qué pasaba:** Si pasábamos los datos al modelo como un array plano de NumPy `model.predict([[1, 120, 70, 20, 80, 25.0, 0.45, 30]])`, la consola se llenaba de advertencias:
  ```text
  UserWarning: X does not have valid feature names, but GaussianNB was fitted with feature names
  ```
- **Cómo lo resolvimos:** Scikit-Learn guarda internamente los nombres de las columnas cuando se entrena con un DataFrame de Pandas (`feature_names_in_`). La solución fue empaquetar siempre los inputs del formulario en un `pd.DataFrame` con los nombres exactos y en el mismo orden que en el dataset original:
  ```python
  df_eval = pd.DataFrame({
      'Pregnancies': [embarazos],
      'Glucose': [glucosa],
      'BloodPressure': [presion],
      'SkinThickness': [pliegue_cutaneo],
      'Insulin': [insulina],
      'BMI': [imc],
      'DiabetesPedigreeFunction': [pedigree_val],
      'Age': [edad]
  })
  prediccion = model.predict(df_eval)[0]
  ```

### Error 3: El motor OCR en entornos Windows vs. Linux (Streamlit Cloud)
- **Qué pasaba:** En nuestros computadores personales con Windows 11 usamos la librería `winocr`, que es espectacular porque se conecta directo al motor OCR nativo del sistema sin configurar nada. Pero al pensar en Streamlit Cloud (que corre en contenedores Linux Debian), sabíamos que `winocr` no existe en Linux.
- **Cómo lo resolvimos:** Encapsulamos la importación y ejecución del OCR dentro de un bloque seguro `try/except Exception`. De esta manera, si la app corre en Windows extrae los datos automáticamente del examen; y si corre en un servidor Linux donde `winocr` no esté presente, la aplicación no colapsa, sino que degrada suavemente permitiendo el ingreso manual de los datos sin interrumpir el funcionamiento del clasificador Naïve Bayes.

### Error 4: Pérdida del estado al interactuar con el formulario (`st.session_state`)
- **Qué pasaba:** En Streamlit, cada vez que el usuario presiona un botón o cambia un valor numérico, el script se vuelve a ejecutar de arriba a abajo. Al inicio, cuando el OCR leía la imagen y colocaba la glucosa en 130, en cuanto el usuario iba a cambiar la edad, la página se recargaba y volvía a poner los valores por defecto, borrando la lectura del examen.
- **Cómo lo resolvimos:** Implementamos el manejo del estado con `st.session_state`:
  ```python
  if 'glucose' not in st.session_state: 
      st.session_state.glucose = 100.0

  # Al detectar valores en la imagen:
  if 'Glucose' in extracted: 
      st.session_state.glucose = float(extracted['Glucose'])
  ```
  De esta forma los valores persisten entre re-renders y la experiencia de usuario se siente fluida y natural.

---

## 10. Despliegue Continuo en la Nube (Streamlit Cloud + GitHub)

Para que el profesor y cualquier evaluador puedan acceder al proyecto sin tener que clonar código ni instalar Python en su máquina, desplegamos la solución en **Streamlit Community Cloud**:

```
[Código en Local] ─── git push ───► [Repositorio GitHub] ─── Webhook ───► [Streamlit Cloud] ───► URL Pública Activa
```

### El proceso paso a paso que seguimos:
1. **Control de versiones con Git:** Creamos el repositorio local y lo enlazamos con GitHub:
   ```bash
   git init
   git add .
   git commit -m "Entrega actividad 3 - Modelo Naive Bayes desplegado en la nube"
   git remote add origin https://github.com/joandestructor777/ia.git
   git push -u origin main
   ```
2. **Conexión con Streamlit Cloud:**
   - Ingresamos a [share.streamlit.io](https://share.streamlit.io/) e iniciamos sesión vinculando la cuenta de GitHub de Joan (`joandestructor777`).
   - Seleccionamos la opción **"New app"** y elegimos el repositorio `joandestructor777/ia` en la rama `main`.
   - En la casilla **Main file path**, definimos la ruta del punto de entrada: `app/app.py`.
3. **Instalación y Despliegue Automático:**
   - Streamlit Cloud detecta el archivo `requirements.txt` en la raíz del repositorio, descarga automáticamente las librerías en un contenedor e inicia el servidor.
   - En menos de dos minutos el modelo queda operativo en una URL pública compartible.
   - **Integración Continua (CI/CD):** Cada vez que hacemos un `git push origin main` con mejoras en el código o en la interfaz, el servidor en la nube se actualiza automáticamente sin tiempos muertos.

---

## 11. Estructura del Repositorio

El proyecto cumple al 100% con la estructura de carpetas estandarizada solicitada en la guía de la materia:

```text
actividad3_naive_bayes/
│
├── README.md                      # Documentación completa, reflexiones y guía del proyecto
├── requirements.txt               # Lista de dependencias y librerías de Python
├── .gitignore                     # Archivos y temporales excluidos del control de versiones
│
├── data/
│   ├── raw/
│   │   ├── dataset.csv            # Dataset original Pima Indians Diabetes
│   │   └── diabetes.csv           # Copia de respaldo de datos crudos
│   └── processed/
│       └── dataset_clean.csv      # Dataset limpio con ceros biológicos imputados por mediana
│
├── notebooks/
│   └── 01_training.ipynb          # Notebook de Jupyter con EDA, partición, entrenamiento y métricas
│
├── models/
│   └── modelo.pkl                 # Modelo GaussianNB entrenado y serializado con Joblib
│
├── src/
│   ├── __init__.py                # Inicializador de módulo de Python
│   ├── preprocessing.py           # Funciones modulares de limpieza e imputación
│   └── predict.py                 # Funciones para cargar modelo.pkl y generar predicciones
│
├── app/
│   ├── .streamlit/
│   │   └── config.toml            # Configuración de tema y servidor de Streamlit
│   ├── app.py                     # Aplicación web principal "Pulso" (rediseño moderno + OCR)
│   ├── app_original_backup.py     # Copia histórica de la primera versión básica de la app
│   └── ejemplo_examen_colombia.png# Imagen de prueba de examen de laboratorio clínico
│
└── docs/
    └── presentacion.pptx          # Presentación ejecutiva en diapositivas para sustentación
```

---

## 12. Cómo Ejecutar el Proyecto en tu Máquina Local

Si deseas clonar el proyecto y correrlo en tu propio computador, sigue estos sencillos pasos:

### 1. Clonar el repositorio
```bash
git clone https://github.com/joandestructor777/ia.git
cd ia
```

### 2. Crear y activar el entorno virtual
```bash
# En Windows:
python -m venv venv
venv\Scripts\activate

# En Linux o macOS:
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar las librerías requeridas
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación web con Streamlit
```bash
streamlit run app/app.py
```

Streamlit abrirá automáticamente la aplicación en tu navegador web predeterminado en la dirección local:
```
http://localhost:8501
```

Allí podrás ingresar los biomarcadores manualmente, usar la cámara para capturar un reporte físico o presionar el botón de **"Examen de prueba (Bogotá)"** para ver el funcionamiento del tamizaje en tiempo real.

---

## 13. Reflexiones Finales y Aprendizajes de la Actividad

Haber desarrollado esta Actividad 3 nos dejó aprendizajes que van mucho más allá de simplemente aprobar un taller universitario:

1. **El valor de cuestionar los datos:** Descubrir que los ceros en variables clínicas eran valores ausentes nos enseñó que la fase de exploración y limpieza de datos (EDA) es donde realmente se gana o se pierde un proyecto de Machine Learning. Si hubiéramos entrenado a ciegas con ceros, el modelo habría aprendido relaciones fisiológicas completamente falsas.
2. **La elegancia de Naïve Bayes:** A pesar de ser un algoritmo que asume una premisa "ingenua" de independencia, Gaussian Naïve Bayes demostró ser extremadamente rápido, interpretable y consistente. Nos permitió entender cómo se combinan probabilidades a priori y verosimilitudes sin cajas negras incomprensibles.
3. **El diseño humano en Inteligencia Artificial:** La IA no vive en el vacío dentro de un notebook de Jupyter. Al construir la interfaz de **Pulso**, aprendimos que cómo se le comunica un resultado a un ser humano es tan importante como el algoritmo en sí. Cambiar alertas rojas de pánico por porcentajes de probabilidad preventiva y sugerir consultas con profesionales de la salud en el marco del sistema médico colombiano le da un propósito ético y real a la tecnología.
4. **Cerrar el ciclo completo (MLOps):** Lograr que un modelo pase de un notebook a un script modular, se guarde como un binario serializado, se integre en una interfaz web y se despliegue de forma continua en la nube nos dio una visión completa de lo que significa construir software con inteligencia artificial en la actualidad.

---

**Autores del Proyecto (Grupo 3):**  
- **Joan Camilo González Camacho** — Estudiante de Ingeniería  
- **Juan David Ruiz Carvajalino** — Estudiante de Ingeniería  
*Materia: Inteligencia Artificial I — 2026*
