from .Cluster import Cluster
import numpy as np

class DBSCAN(Cluster):
    def __init__(self, eps=0.5, min_samples=5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                continue
            self._expand_cluster(X, labels, i, neighbors, cluster_id, visited)
            cluster_id += 1

        self.labels_ = labels
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    def _region_query(self, X, idx):
        return np.where(np.linalg.norm(X - X[idx], axis=1) <= self.eps)[0]

    def _expand_cluster(self, X, labels, idx, neighbors, cluster_id, visited):
        labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            n_idx = neighbors[i]
            if not visited[n_idx]:
                visited[n_idx] = True
                new_neighbors = self._region_query(X, n_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            if labels[n_idx] == -1:
                labels[n_idx] = cluster_id
            i += 1

    def _assign_labels(self, X):
        return self.labels_
