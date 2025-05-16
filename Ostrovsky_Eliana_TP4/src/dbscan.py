import numpy as np

def encontrar_vecinos(X, idx, eps):
    """Devuelve los índices de los puntos dentro de eps del punto X[idx]."""
    distancias = np.linalg.norm(X - X[idx], axis=1)
    return np.where(distancias <= eps)[0]

def expandir_cluster(X, etiquetas, punto_idx, cluster_id, eps, min_pts):
    """Expande un nuevo cluster si el punto es un núcleo."""
    vecinos = encontrar_vecinos(X, punto_idx, eps)

    if len(vecinos) < min_pts:
        etiquetas[punto_idx] = -1  # Ruido
        return False
    else:
        etiquetas[punto_idx] = cluster_id
        i = 0
        while i < len(vecinos):
            vecino_idx = vecinos[i]
            if etiquetas[vecino_idx] == -1:
                etiquetas[vecino_idx] = cluster_id  # Reclasificar ruido como parte del cluster
            elif etiquetas[vecino_idx] == 0:
                etiquetas[vecino_idx] = cluster_id
                vecinos_nuevos = encontrar_vecinos(X, vecino_idx, eps)
                if len(vecinos_nuevos) >= min_pts:
                    vecinos = np.concatenate((vecinos, vecinos_nuevos))
            i += 1
        return True

def dbscan(X, eps, min_pts):
    """
    Algoritmo DBSCAN desde cero.
    Etiquetas: -1 = ruido, 1...K = id de cluster.
    """
    n = X.shape[0]
    etiquetas = np.zeros(n, dtype=int)
    cluster_id = 0

    for i in range(n):
        if etiquetas[i] != 0:
            continue
        if expandir_cluster(X, etiquetas, i, cluster_id + 1, eps, min_pts):
            cluster_id += 1

    return etiquetas
