# Despliegue de Modelo de Machine Learning en la Nube - Actividad 3

Este es mi proyecto para la Actividad 3 de Inteligencia Artificial I. En la actividad pasada trabajé con PCA y para esta entrega el profesor me reasignó el algoritmo **Naive Bayes**. Desarrollé un modelo para predecir si una persona puede tener diabetes según sus exámenes médicos y creé una aplicación web en Streamlit para ponerlo a funcionar en la nube.

---

## 1. Problema y Datos Reales
- **Problema:** Predicción de diabetes a partir de variables médicas. Me parece un problema práctico y fácil de demostrar en pocos minutos.
- **Dataset:** Utilicé el dataset de *Pima Indians Diabetes*, que tiene 768 registros reales y 8 variables (glucosa, presión, insulina, IMC, edad, etc.).
- **Limpieza de datos:** En lo que vi en los datos, variables como glucosa o presión venían con valor 0, lo cual no tiene sentido en una persona viva. Me pareció mejor reemplazar esos ceros por la mediana para no perder filas.

---

## 2. El Algoritmo: Naive Bayes
Para este proyecto usé **Gaussian Naive Bayes** (`GaussianNB`) de la librería scikit-learn. 

Básicamente, lo que hace el algoritmo es calcular probabilidades con el Teorema de Bayes para ver a qué clase pertenece cada paciente:
- Calcula qué tan común es la diabetes en los datos.
- Mira el valor de cada variable del paciente (como su glucosa o edad) y calcula la probabilidad de que pertenezca al grupo con diabetes o al grupo sano.
- Al final multiplica esas probabilidades y elige la que dé mayor puntaje.

Me parece interesante este algoritmo porque es rápido de entrenar y funciona bien sin pedir configuraciones raras.

---

## 3. Estructura del Repositorio
El proyecto está organizado exactamente como pide la guía:

```text
actividad3_naive_bayes/
│
├── README.md              # Esta documentación
├── requirements.txt       # Librerías necesarias
├── .gitignore             # Archivos a ignorar en git
│
├── data/
│   ├── raw/               # Dataset original (dataset.csv)
│   └── processed/         # Dataset limpio (dataset_clean.csv)
│
├── notebooks/
│   └── 01_training.ipynb  # Notebook con el paso a paso del entrenamiento
│
├── models/
│   └── modelo.pkl         # Modelo entrenado y guardado
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # Función para limpiar datos
│   └── predict.py         # Función para hacer predicciones con el modelo
│
├── app/
│   └── app.py             # Aplicación web hecha con Streamlit
│
└── docs/
    └── presentacion.pdf   # Presentación final para la sustentación
```

---

## 4. Cómo Probar la Aplicación en tu Computador

1. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecutar la aplicación web:
```bash
streamlit run app/app.py
```

La aplicación se abre sola en el navegador en `http://localhost:8501`. Ahí se pueden ingresar los datos del paciente y al presionar "Predecir" muestra el resultado en pantalla.

---

## 5. Despliegue en la Nube (Streamlit Cloud)

Para poner la aplicación en la nube usé **Streamlit Cloud** porque es gratuito y se conecta directo con GitHub:

1. Subir esta carpeta a un repositorio público de GitHub.
2. Entrar a [share.streamlit.io](https://share.streamlit.io/) e iniciar sesión con la cuenta de GitHub.
3. Darle al botón "New app", seleccionar el repositorio y en la ruta del archivo poner: `app/app.py`.
4. Darle a "Deploy" y en un par de minutos ya queda la URL pública funcionando.

---

## 6. Integrantes (Grupo 3)
- Joan Camilo Gonzalez Camacho
- Juan David Ruiz Carvajalino
