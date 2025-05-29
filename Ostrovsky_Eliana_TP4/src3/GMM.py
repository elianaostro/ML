import numpy as np
from Cluster import Cluster
from KMeans import KMeans

class GMM(Cluster):
    def __init__(self, n_clusters=1, max_iter=100, tol=1e-3, random_state=None):
        super().__init__(random_state)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.converged_ = False

    def fit(self, X):
        n_samples, n_features = X.shape
        K = self.n_clusters

        # indices = np.random.choice(n_samples, K, replace=False)
        # self.means_ = X[indices]
        # self.weights_ = np.ones(K) / K
        # self.covariances_ = np.array([np.cov(X, rowvar=False) for _ in range(K)])

        model_km = KMeans(n_clusters=K, random_state=self.random_state)
        model_km.fit(X)
        indices = model_km.labels_
        self.means_ = model_km.centroids_
        self.weights_ = np.bincount(indices, minlength=K) / n_samples
        self.covariances_ = np.array([
            np.cov(X[indices == k], rowvar=False) + 1e-6 * np.eye(n_features)
            for k in range(K)
        ])

        log_likelihood_old = None

        for _ in range(self.max_iter):
            resp = np.zeros((n_samples, K))
            for k in range(K):
                resp[:, k] = self.weights_[k] * self._gaussian(X, self.means_[k], self.covariances_[k])
            resp /= resp.sum(axis=1, keepdims=True)

            Nk = resp.sum(axis=0)
            self.weights_ = Nk / n_samples
            self.means_ = (resp.T @ X) / Nk[:, None]
            self.covariances_ = np.array([
                (resp[:, k][:, None] * (X - self.means_[k])).T @ (X - self.means_[k]) / Nk[k]
                for k in range(K)
            ])

            log_likelihood = np.sum(np.log(np.sum([
                self.weights_[k] * self._gaussian(X, self.means_[k], self.covariances_[k])
                for k in range(K)
            ], axis=0)))

            if log_likelihood_old is not None and abs(log_likelihood - log_likelihood_old) < self.tol:
                self.converged_ = True
                break
            log_likelihood_old = log_likelihood

        self.labels_ = self._assign_labels(X)

    def _gaussian(self, X, mean, cov):
        n = X.shape[1]
        cov_inv = np.linalg.inv(cov)
        diff = X - mean
        exponent = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
        denom = np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))
        return np.exp(-0.5 * exponent) / denom

    def _assign_labels(self, X):
        probs = np.array([
            self.weights_[k] * self._gaussian(X, self.means_[k], self.covariances_[k])
            for k in range(self.n_clusters)
        ]).T
        return np.argmax(probs, axis=1)

    def predict(self, X):
        return self._assign_labels(X)
