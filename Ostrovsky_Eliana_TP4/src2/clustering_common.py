import numpy as np

def inicializar_centroides(X, k, seed=0):
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def asignar_por_distancia(X, centroides):
    distancias = np.linalg.norm(X[:, np.newaxis] - centroides, axis=2)
    return np.argmin(distancias, axis=1)

def calcular_inercia(X, centroides, labels):
    return sum(np.linalg.norm(X[i] - centroides[labels[i]]) ** 2 for i in range(len(X)))

def pdf_gaussiana(x, mu, sigma):
    d = x.shape[0]
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    norm = 1 / np.sqrt((2 * np.pi) ** d * det)
    diff = x - mu
    exponent = -0.5 * diff.T @ inv @ diff
    return norm * np.exp(exponent)
