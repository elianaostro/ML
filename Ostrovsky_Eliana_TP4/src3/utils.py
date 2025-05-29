import matplotlib.pyplot as plt
import numpy as np
import KMeans
import GMM
import DBSCAN
from sklearn.metrics import silhouette_score

def plot_elbow(X, k_range, method='kmeans', random_state=None):
    """
    Método del codo: para KMeans calcula la suma de distancias mínimas (L),
    y para GMM usa la log-verosimilitud negativa (-log-likelihood).

    Parameters:
        - X: array de datos
        - k_range: lista de valores de K a evaluar
        - method: 'kmeans' o 'gmm'
        - random_state: semilla opcional

    Devuelve:
        - lista de pérdidas (L o -loglikelihood)
    """
    losses = []
    if method not in ('kmeans', 'gmm'):
        raise ValueError("El método debe ser 'kmeans' o 'gmm'.")

    for k in k_range:
        if method == 'kmeans':
            model = KMeans(n_clusters=k, random_state=random_state)
            model.fit(X)
            centroids = model.centroids_
        else:
            model = GMM(n_components=k, random_state=random_state)
            model.fit(X)
            centroids = model.means_
        L = np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1))
        losses.append(L)

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, losses, marker='o')
    plt.xlabel('Cantidad de clusters (K)')
    ylabel = 'Suma de distancias (KMeans)' if method == 'kmeans' else '-Log-verosimilitud (GMM)'
    plt.ylabel(ylabel)
    plt.title(f'Método del codo ({method.upper()})')
    plt.grid(True)
    plt.show()

    return losses

def plot_clusters(X, labels, centroids=None, title="Clusters"):
    """
    Dibuja los puntos de datos coloreados por etiqueta, y los centroides si existen.
    """
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        if label == -1:
            color = 'k'
            marker = 'x'
            label_name = 'Ruido'
        else:
            color = colors(i)
            marker = 'o'
            label_name = f'Cluster {label}'
        
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], c=[color], label=label_name, marker=marker)

    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='*', label='Centroides')

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def silhouette_score_manual(X, labels):
    """
    Calcula el silhouette score promedio manualmente.
    """
    n = len(X)
    unique_labels = set(labels)
    if len(unique_labels) <= 1 or -1 in unique_labels and len(unique_labels) == 2:
        return -1  
    
    silhouette_scores = []

    for i in range(n):
        xi = X[i]
        label_i = labels[i]
        if label_i == -1:
            continue

        same_cluster = [j for j in range(n) if labels[j] == label_i and j != i]
        other_clusters = [l for l in unique_labels if l != label_i and l != -1]

        if not same_cluster:
            a = 0
        else:
            a = np.mean([np.linalg.norm(xi - X[j]) for j in same_cluster])

        b_values = []
        for other_label in other_clusters:
            other_points = [j for j in range(n) if labels[j] == other_label]
            if other_points:
                b = np.mean([np.linalg.norm(xi - X[j]) for j in other_points])
                b_values.append(b)

        if not b_values:
            continue

        b = min(b_values)
        s = (b - a) / max(a, b)
        silhouette_scores.append(s)

    return np.mean(silhouette_scores) if silhouette_scores else -1


def explore_dbscan_params(X, eps_values, min_samples_values):
    """
    Evalúa combinaciones de parámetros para DBSCAN usando silhouette score manual.
    """
    best_score = -1
    best_params = None
    scores = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            model.fit(X)
            labels = model.labels_

            if len(set(labels)) > 1:
                score = silhouette_score_manual(X, labels)
                scores.append((eps, min_samples, score))
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)

    if best_params:
        print(f"Mejores parámetros: eps={best_params[0]}, min_samples={best_params[1]} con silhouette={best_score:.3f}")
    else:
        print("No se encontró una combinación válida con más de un cluster.")

    return scores, best_params
