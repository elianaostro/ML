import numpy as np

def vecinos(X, i, eps):
    dists = np.linalg.norm(X - X[i], axis=1)
    return np.where(dists <= eps)[0]

def expandir(X, etiquetas, i, cluster_id, eps, min_pts):
    vecinos_i = vecinos(X, i, eps)
    if len(vecinos_i) < min_pts:
        etiquetas[i] = -1
        return False
    etiquetas[i] = cluster_id
    i_vecinos = list(vecinos_i)
    j = 0
    while j < len(i_vecinos):
        p = i_vecinos[j]
        if etiquetas[p] == -1:
            etiquetas[p] = cluster_id
        elif etiquetas[p] == 0:
            etiquetas[p] = cluster_id
            nuevos = vecinos(X, p, eps)
            if len(nuevos) >= min_pts:
                i_vecinos.extend(nuevos)
        j += 1
    return True

def dbscan(X, eps, min_pts):
    n = len(X)
    etiquetas = np.zeros(n, dtype=int)
    cluster_id = 0
    for i in range(n):
        if etiquetas[i] != 0:
            continue
        if expandir(X, etiquetas, i, cluster_id + 1, eps, min_pts):
            cluster_id += 1
    return etiquetas
