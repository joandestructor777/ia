import pandas as pd
import numpy as np

# Columnas donde el valor 0 es un dato faltante
columnas_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def limpiar_datos(df):
    # Reemplaza los ceros por NaN y luego por la mediana
    df_limpio = df.copy()
    for col in columnas_ceros:
        if col in df_limpio.columns:
            df_limpio[col] = df_limpio[col].replace(0, np.nan)
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
    return df_limpio
