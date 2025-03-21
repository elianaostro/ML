import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# Agregar el path de los modelos
sys.path.append(os.path.abspath(".."))
from src import models  # Importar modelo de regresión

# Cargar datos
def cargar_datos(filepath):
    print(f"\nCargando datos desde: {filepath}")
    return pd.read_csv(filepath)

# Preprocesamiento de datos
def preprocesar_datos(df):
    print("\nPreprocesando datos...")

    # Convertir 'area' a metros cuadrados si está en sqft
    df.loc[df["area_units"] == "sqft", "area"] *= 0.092903
    df.drop(columns=["area_units"], inplace=True)

    # Imputación de habitaciones según área
    df["rooms"].fillna(df.groupby("area")["rooms"].transform("median"), inplace=True)

    # Imputación de edad por mediana
    df["age"].fillna(df.groupby("is_house")["age"].transform("median"), inplace=True)

    return df

# Normalización
def normalizar_datos(X_train, X_val, columnas):
    print("\nNormalizando datos...")
    medianas = X_train[columnas].median()
    desviaciones = X_train[columnas].std()
    
    X_train[columnas] = (X_train[columnas] - medianas) / desviaciones
    X_val[columnas] = (X_val[columnas] - medianas) / desviaciones

    return X_train, X_val, medianas, desviaciones

# Entrenar modelo de regresión
def entrenar_modelo(X_train, y_train, metodo="pseudoinversa"):
    print(f"\nEntrenando modelo con {metodo}...")
    modelo = models.LinearRegression(X_train, y_train)
    if metodo == "pseudoinversa":
        modelo.train_pseudoinverse()
    elif metodo == "descenso_gradiente":
        modelo.train_gradient_descent(lr=0.01, epochs=1000)
    return modelo

# Evaluar modelo
def evaluar_modelo(modelo, X, y, nombre="validación"):
    print(f"\nEvaluando modelo en {nombre}...")
    y_pred = modelo.predict(X)
    error = models.mse(y, y_pred)
    print(f"Error cuadrático medio en {nombre}: {error:.4f}")
    return error

# Configuración de dataset
df = cargar_datos("../data/raw/casas_dev.csv")
df = preprocesar_datos(df)

# Dividir en train y validación
train_size = int(0.8 * len(df))
X_train, X_val = df.iloc[:train_size].drop(columns=["price"]), df.iloc[train_size:].drop(columns=["price"])
y_train, y_val = df.iloc[:train_size]["price"].values, df.iloc[train_size:]["price"].values

# Columnas a normalizar
columnas_norm = ["area", "age", "rooms"]
X_train, X_val, medianas, desviaciones = normalizar_datos(X_train, X_val, columnas_norm)

# Entrenamiento y evaluación
modelo = entrenar_modelo(X_train, y_train)
evaluar_modelo(modelo, X_train, y_train, "train")
evaluar_modelo(modelo, X_val, y_val, "validación")

# Imprimir coeficientes
modelo.print_coefficients(X_train.columns)

# Predecir datos de Amanda
df_amanda = cargar_datos("../data/raw/vivienda_Amanda.csv")
df_amanda = preprocesar_datos(df_amanda)
df_amanda[columnas_norm] = (df_amanda[columnas_norm] - medianas) / desviaciones

X_amanda = df_amanda[["area", "is_house", "has_pool", "age", "lat", "lon"]]
df_amanda["predicted_price"] = modelo.predict(X_amanda)
print("\nPredicciones para vivienda Amanda:")
print(df_amanda.head())

# Valor por metro cuadrado de una casa
casas = df[df["is_house"] == 1]
valor_prom_m2 = (casas["price"] / casas["area"]).mean()
print(f"\nValor promedio por metro cuadrado de una casa: ${valor_prom_m2:.2f}")

# Impacto de la pileta en el precio
coef = modelo.coef.flatten()
impacto_pileta = coef[list(X_train.columns).index("has_pool") + 1] * desviaciones["has_pool"]
print(f"\nImpacto estimado de construir una pileta: ${impacto_pileta:.2f}")
