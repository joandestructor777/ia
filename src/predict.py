import os
import joblib
import pandas as pd

def cargar_modelo():
    # Carga el modelo serializado
    ruta = os.path.join(os.path.dirname(__file__), '..', 'models', 'modelo.pkl')
    return joblib.load(ruta)

def predecir(df_datos):
    # Hace la prediccion con el modelo
    modelo = cargar_modelo()
    prediccion = modelo.predict(df_datos)
    probabilidades = modelo.predict_proba(df_datos)
    return prediccion[0], probabilidades[0]
