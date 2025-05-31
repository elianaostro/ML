import numpy as np
from itertools import product
from tqdm import tqdm


def elbow_method(X, k_range, model, random_state=None): 
    """
    Método del codo: para KMeans calcula la suma de distancias mínimas (L),
    y para GMM usa la log-verosimilitud negativa (-log-likelihood).

    Parameters:
        - X: array de datos
        - k_range: lista de valores de K a evaluar
        - model: modelo a utilizar
        - random_state: semilla opcional

    Devuelve:
        - lista de pérdidas (L o -loglikelihood)
    """
    losses = []

    for k in k_range:
        modelo = model(n_clusters=k, random_state=random_state)
        modelo.fit(X)
        centroids = modelo.means_
        L = np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1))
        losses.append(L)

    return losses


def fit_random_seed(model, X, k, seed_range=range(100)):
    """
    Ajusta el modelo con diferentes semillas y devuelve el índice de la mejor semilla
    según la suma de distancias mínimas (L) para KMeans o la log-verosimilitud negativa para GMM.
    Parameters:
        - model: modelo a utilizar (KMeans o GMM)
        - X: datos de entrada
        - k: número de clusters
        - seed_range: rango de semillas a evaluar (default: range(100))
    Returns:
        - índice de la mejor semilla
    """
    losses = []
    for i in seed_range:
        np.random.seed(i)
        modelo = model(n_clusters=k, random_state=i)
        modelo.fit(X)
        L = np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - modelo.means_, axis=2), axis=1))
        losses.append(L)

    return np.argmin(losses)


def silhouette_score(X, labels, dists=None):
    """
    Calcula el silhouette score promedio manualmente, optimizado con matriz de distancias.
    """
    n = len(X)
    unique_labels = set(labels)
    if len(unique_labels) <= 1 or (-1 in unique_labels and len(unique_labels) == 2):
        return -1

    if dists is None:
        dists = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)

    silhouette_scores = []

    for i in range(n):
        label_i = labels[i]
        if label_i == -1:
            continue

        same_cluster = np.where((labels == label_i) & (np.arange(n) != i))[0]
        other_labels = [l for l in unique_labels if l != label_i and l != -1]

        a = np.mean(dists[i][same_cluster]) if len(same_cluster) > 0 else 0

        b_values = [
            np.mean(dists[i][labels == other_label])
            for other_label in other_labels
            if np.any(labels == other_label)
        ]

        if not b_values:
            continue

        b = min(b_values)
        s = (b - a) / max(a, b)
        silhouette_scores.append(s)

    return np.mean(silhouette_scores) if silhouette_scores else -1


def penalized_silhouette_score(X, labels, dists=None):
    """
    Calcula un silhouette score penalizado por la cantidad de puntos de ruido (-1).
    """
    n = len(X)
    n_noise = np.sum(labels == -1)
    noise_ratio = n_noise / n

    base_score = silhouette_score(X, labels, dists=dists)

    if base_score == -1:
        return -1

    penalty = 1 - noise_ratio  
    penalized_score = base_score * penalty

    return penalized_score


def explore_dbscan_params(X, eps_values, min_samples_values):
    """
    Evalúa combinaciones de parámetros para DBSCAN usando silhouette score optimizado.
    """
    from src.DBSCAN import DBSCAN
    
    best_score = -1
    best_params = None
    scores = []

    dists = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)

    param_combinations = list(product(eps_values, min_samples_values))
    for eps, min_samples in tqdm(param_combinations, desc="Evaluando DBSCAN..."):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(X)
        labels = model.labels_

        score = penalized_silhouette_score(X, labels, dists=dists)

        if score != -1:
            scores.append((eps, min_samples, score))
            if score > best_score:
                best_score = score
                best_params = (eps, min_samples)

    if best_params:
        print(f"Mejores parámetros: eps={best_params[0]}, min_samples={best_params[1]} con silhouette={best_score:.3f}")
    else:
        print("No se encontró una combinación válida con más de un cluster.")

    return scores, best_params
