import streamlit as st
import joblib
import pandas as pd
import os

# Titulo de la aplicacion
st.title('Predictor de Diabetes con Naive Bayes')
st.write('Esta aplicación permite predecir el riesgo de diabetes en un paciente usando un modelo de Naive Bayes entrenado.')

# Carga del modelo
ruta_modelo = os.path.join(os.path.dirname(__file__), '..', 'models', 'modelo.pkl')
model = joblib.load(ruta_modelo)

# Inputs del usuario con las 8 caracteristicas
st.subheader('Ingrese los datos del paciente:')

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Número de Embarazos', min_value=0, max_value=20, value=1)
    glucose = st.number_input('Glucosa en sangre (mg/dL)', min_value=0.0, max_value=300.0, value=100.0)
    blood_pressure = st.number_input('Presión Arterial Diastólica (mm Hg)', min_value=0.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input('Grosor del pliegue de piel (mm)', min_value=0.0, max_value=100.0, value=20.0)

with col2:
    insulin = st.number_input('Insulina en suero (µU/mL)', min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input('Índice de Masa Corporal - IMC (kg/m²)', min_value=0.0, max_value=70.0, value=25.0)
    pedigree = st.number_input('Función de Pedigrí de Diabetes', min_value=0.0, max_value=3.0, value=0.47)
    age = st.number_input('Edad (años)', min_value=1, max_value=120, value=30)

# Boton de prediccion
if st.button('Predecir'):
    # Preparar el dataframe de entrada con los mismos nombres del dataset
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [pedigree],
        'Age': [age]
    })
    
    # Realizar la prediccion
    prediccion = model.predict(input_data)[0]
    probabilidades = model.predict_proba(input_data)[0]
    
    # Mostrar el resultado
    st.subheader('Resultado:')
    if prediccion == 1:
        st.error(f'Predicción: Positivo para Diabetes (Probabilidad: {probabilidades[1]*100:.1f}%)')
        st.write('En lo que veo, los valores indican un riesgo alto. Se recomienda consultar al médico.')
    else:
        st.success(f'Predicción: Negativo para Diabetes (Probabilidad: {probabilidades[0]*100:.1f}%)')
        st.write('Me parece que los valores están dentro de lo normal, pero siempre es bueno mantener hábitos saludables.')
