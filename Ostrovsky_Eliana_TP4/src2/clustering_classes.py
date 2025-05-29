import numpy as np
from abc import ABC, abstractmethod
import sys

class ClusteringBase(ABC):
    """Clase base para algoritmos de clustering que comparten funcionalidades comunes."""
    
    def __init__(self, k=None, max_iter=100, tol=1e-4, seed=0):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroides_ = None
        self.labels_ = None
        self.is_fitted_ = False
    
    def inicializar_centroides(self, X, k=None, seed=None):
        """Inicializa centroides aleatoriamente."""
        k = k or self.k
        seed = seed if seed is not None else self.seed
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.choice(len(X), k, replace=False)
        return X[indices]
    
    def asignar_por_distancia(self, X, centroides=None):
        """Asigna cada punto al centroide más cercano."""
        centroides = centroides if centroides is not None else self.centroides_
        distancias = np.linalg.norm(X[:, np.newaxis] - centroides, axis=2)
        return np.argmin(distancias, axis=1)
    
    def calcular_inercia(self, X, centroides=None, labels=None):
        """Calcula la inercia (suma de distancias cuadráticas)."""
        centroides = centroides if centroides is not None else self.centroides_
        labels = labels if labels is not None else self.labels_
        return sum(np.linalg.norm(X[i] - centroides[labels[i]]) ** 2 for i in range(len(X)))
    
    def pdf_gaussiana(self, x, mu, sigma):
        """Calcula la densidad de probabilidad gaussiana."""
        d = x.shape[0]
        det = np.linalg.det(sigma)
        inv = np.linalg.inv(sigma)
        norm = 1 / np.sqrt((2 * np.pi) ** d * det)
        diff = x - mu
        exponent = -0.5 * diff.T @ inv @ diff
        return norm * np.exp(exponent)
    
    def update_progress_bar(self, current_iter, total_iters, bar_length=50, metrics=None):
        """Muestra barra de progreso durante el entrenamiento."""
        percent = float(current_iter) / total_iters
        arrow_len = max(1, int(round(percent * bar_length)))
        arrow = '=' * (arrow_len - 1) + '>' if arrow_len > 1 else '>'
        spaces = ' ' * (bar_length - arrow_len)
        
        metrics_str = ""
        if metrics:
            metrics_str = " - " + " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        sys.stdout.write(f"\rIteración: {current_iter}/{total_iters} [{arrow + spaces}] {int(percent * 100)}%{metrics_str}")
        sys.stdout.flush()
        if current_iter == total_iters:
            print()
    
    @abstractmethod
    def fit(self, X):
        """Método abstracto para entrenar el modelo."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Método abstracto para predecir clusters."""
        pass
    
    def fit_predict(self, X):
        """Entrena el modelo y devuelve las predicciones."""
        self.fit(X)
        return self.predict(X)