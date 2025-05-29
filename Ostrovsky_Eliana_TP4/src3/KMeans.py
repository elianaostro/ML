from .Cluster import Cluster
import numpy as np

class KMeans(Cluster):
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        super().__init__(random_state)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                for i in range(self.n_clusters)
            ])

            if np.linalg.norm(new_centroids - centroids) < self.tol:
                break
            centroids = new_centroids

        self.centroids_ = centroids
        self.labels_ = labels

    def predict(self, X):
        return self._assign_labels(X)

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2)
        return np.argmin(distances, axis=1)
