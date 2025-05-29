import numpy as np
from abc import ABC, abstractmethod

class Cluster(ABC):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.labels_ = None
        self.n_clusters = None
        if random_state is not None:
            np.random.seed(random_state)

    @abstractmethod
    def fit(self, X): pass

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        raise NotImplementedError("Este método debe ser implementado por el algoritmo.")

    @abstractmethod
    def _assign_labels(self, X): pass
