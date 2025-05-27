import numpy as np
from src.clustering_common import inicializar_centroides, asignar_por_distancia, calcular_inercia

def actualizar_centroides(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def kmeans(X, k, max_iter=100, tol=1e-4, seed=0):
    centroides = inicializar_centroides(X, k, seed=seed)
    for _ in range(max_iter):
        labels = asignar_por_distancia(X, centroides)
        nuevos_centroides = actualizar_centroides(X, labels, k)
        if np.allclose(centroides, nuevos_centroides, atol=tol):
            break
        centroides = nuevos_centroides
    inercia = calcular_inercia(X, centroides, labels)
    return centroides, labels, inercia

def iter_kmeans(X, k, max_iter=100, tol=1e-4, tries=100):
    best_inercia = np.inf
    best_result = None
    for _ in range(tries):
        resultado = kmeans(X, k, max_iter=max_iter, tol=tol, seed=None)
        if resultado[2] < best_inercia:
            best_inercia = resultado[2]
            best_result = resultado
    return best_result