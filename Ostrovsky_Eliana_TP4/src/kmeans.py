import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def inicializar_centroides(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Selecciona k puntos aleatorios como centroides iniciales."""
    np.random.seed(seed)
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def asignar_clusters(X: np.ndarray, centroides: np.ndarray) -> np.ndarray:
    """Asigna cada punto al cluster cuyo centroide está más cerca."""
    distancias = np.linalg.norm(X[:, np.newaxis] - centroides, axis=2)
    return np.argmin(distancias, axis=1)

def actualizar_centroides(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Calcula nuevos centroides como el promedio de los puntos asignados a cada cluster."""
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def calcular_inercia(X: np.ndarray, centroides: np.ndarray, labels: np.ndarray) -> float:
    """Calcula la suma de distancias cuadradas de cada punto a su centroide."""
    return sum(np.linalg.norm(X[i] - centroides[labels[i]]) ** 2 for i in range(len(X)))

def kmeans(X: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-4) -> tuple:
    """Ejecuta el algoritmo de K-means."""
    centroides = inicializar_centroides(X, k)
    for _ in range(max_iter):
        labels = asignar_clusters(X, centroides)
        nuevos_centroides = actualizar_centroides(X, labels, k)
        if np.all(np.linalg.norm(nuevos_centroides - centroides, axis=1) < tol):
            break
        centroides = nuevos_centroides
    inercia = calcular_inercia(X, centroides, labels)
    return centroides, labels, inercia

def metodo_del_codo(X: np.ndarray,k_min = 2, k_max: int = 10):
    """Ejecuta K-means para distintos valores de K y grafica el método del codo."""
    inercias = []
    for k in range(k_min, k_max + 1):
        centroides, labels, inercia = kmeans(X, k)
        inercias.append({'k': k, 'centroide': centroides, 'labels':labels, 'inercia': inercia})

    return inercias

