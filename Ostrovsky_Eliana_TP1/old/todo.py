import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.hiperparams import find_best_lambda_cross_validation, find_best_lambda_validation, optimal_lambda, plot_regularization_coefficients
from Ostrovsky_Eliana_TP1.src.preprocessing1 import add_relative_location, area_x_price, generate_polynomial_features
from src.utils import estimate, inicialite, try_model

def main():
    df = inicialite()
    print('Modelo 1')
    try_model(df, df.columns.difference(['price']))

    print('Modelo con features adicionales')
    df_2 = area_x_price(add_relative_location(df))
    try_model(df_2, ['area_1', 'area_0','rooms', 'is_house', 'has_pool', 'age', 'lat_1', 'lat_0', 'lon_1', 'lon_0','area_units'])

    print('Modelo con features polinómicas')
    df_3 = generate_polynomial_features(df)
    try_model(df_3, df_3.columns.difference(['price']))

    lambdas = [0.01, 0.1, 1, 10, 100]

    plot_regularization_coefficients(df_2, lambdas, reg_type="ridge")
    plot_regularization_coefficients(df_2, lambdas, reg_type="lasso")

    optimal_lambda(df_2)

    lambdas = np.logspace(-3, 3, 10) 

    best_lambda_val, mse_values_val = find_best_lambda_validation(df, lambdas)
    print(f"Mejor lambda (validación simple): {best_lambda_val}")

    best_lambda_cv, mse_values_cv = find_best_lambda_cross_validation(df, lambdas, k=5)
    print(f"Mejor lambda (validación cruzada): {best_lambda_cv}")


if __name__ == "__main__":
    main()


    import numpy as np
from math import radians, sin, cos, sqrt, atan2

def kmeans_barrios(X, n_clusters=2, max_iter=100, tol=1e-4):
    """
    Implementa K-Means desde cero usando NumPy para encontrar barrios en base a latitud y longitud.

    Parámetros:
    - X: Array de forma (n, 2) con las coordenadas [lat, lon].
    - n_clusters: Número de barrios a identificar.
    - max_iter: Número máximo de iteraciones.
    - tol: Tolerancia para la convergencia.

    Retorna:
    - labels: Array de etiquetas de barrio asignadas a cada propiedad.
    - centroides: Array de coordenadas de los centroides de los barrios.
    """
    # Inicializar centroides aleatoriamente
    np.random.seed(42)
    centroides = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    for _ in range(max_iter):
        # Calcular distancia euclidiana a cada centroide
        distancias = np.linalg.norm(X[:, np.newaxis] - centroides, axis=2)
        
        # Asignar cada punto al centroide más cercano
        labels = np.argmin(distancias, axis=1)
        
        # Recalcular centroides sin eliminar clusters vacíos
        nuevos_centroides = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroides[i]
            for i in range(n_clusters)
        ])
        
        # Verificar convergencia
        if np.linalg.norm(nuevos_centroides - centroides) < tol:
            break
        
        centroides = nuevos_centroides

    return labels, centroides

def calcular_distancia_haversine(lat1, lon1, lat2, lon2):
    """Calcula la distancia Haversine entre dos puntos geográficos en km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 6371 * c  # Radio de la Tierra en km

def caracterizar_barrios(df, n_clusters=2):
    """
    Identifica barrios con K-Means y calcula la distancia de cada propiedad a su centroide.

    Parámetros:
    - df: DataFrame con columnas 'lat' y 'lon'.
    - n_clusters: Número de barrios a detectar.

    Retorna:
    - df con nuevas features 'barrio' y 'distancia_centro_barrio'.
    - centroides de los barrios detectados.
    """
    X = df[["lat", "lon"]].values
    labels, centroides = kmeans_barrios(X, n_clusters)

    # Asignar barrios al DataFrame
    df["barrio"] = labels

    # Calcular distancia al centro del barrio
    df["distancia_centro_barrio"] = [
        calcular_distancia_haversine(lat, lon, centroides[b][0], centroides[b][1])
        for lat, lon, b in zip(df["lat"], df["lon"], labels)
    ]

    return df, centroides

# Aplicar la función al dataset con 2 barrios
df, centroides = caracterizar_barrios(df, n_clusters=2)

# Mostrar los primeros resultados
df.head(), centroides

import numpy as np
from scipy.spatial import KDTree

# Feature 1: Densidad de área por habitación
df["area_per_room"] = df["area"] / df["rooms"]

# Feature 2: Cantidad de vecinos dentro de un radio definido
def count_neighbors(df, radius=0.01):
    """
    Calcula cuántas propiedades tienen vecinos dentro de un radio definido.
    Usa un KDTree para eficiencia en la búsqueda de vecinos cercanos.
    
    Parámetros:
    - df: DataFrame con columnas 'lat' y 'lon'.
    - radius: Radio en unidades de latitud/longitud para contar vecinos.
    
    Retorna:
    - Lista con la cantidad de vecinos por cada propiedad.
    """
    coords = df[["lat", "lon"]].values
    tree = KDTree(coords)
    neighbor_counts = tree.query_ball_point(coords, r=radius)
    
    # Restamos 1 porque cada punto se cuenta a sí mismo
    return [len(neighbors) - 1 for neighbors in neighbor_counts]

# Agregar la nueva feature al dataset
df["num_neighbors"] = count_neighbors(df, radius=0.01)

# Mostrar las primeras filas con las nuevas características
df.head()
