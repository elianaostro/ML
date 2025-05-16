import numpy as np
from src.kmeans import kmeans

def inicializar_gmm(X: np.ndarray, k: int):
    """Inicializa medias, covarianzas y pesos con K-means."""
    medias, labels, _ = kmeans(X, k)
    n = X.shape[0]
    d = X.shape[1]

    # Inicialización de los pesos π_k
    pesos = np.array([np.mean(labels == i) for i in range(k)])

    # Inicialización de las matrices de covarianza
    covarianzas = []
    for i in range(k):
        puntos = X[labels == i]
        if len(puntos) > 1:
            cov = np.cov(puntos.T) + 1e-6 * np.eye(d)  # Estabilización numérica
        else:
            cov = np.eye(d)
        covarianzas.append(cov)
    
    return medias, covarianzas, pesos

def pdf_multivariada(x, media, cov):
    """Calcula la probabilidad de una gaussiana multivariada."""
    d = x.shape[0]
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * cov_det)
    diff = x - media
    exponent = -0.5 * diff.T @ cov_inv @ diff
    return norm_const * np.exp(exponent)

def expectation_step(X, medias, covarianzas, pesos):
    """Calcula las responsabilidades (matriz gamma)."""
    n = X.shape[0]
    k = len(medias)
    gamma = np.zeros((n, k))
    
    for i in range(n):
        for j in range(k):
            gamma[i, j] = pesos[j] * pdf_multivariada(X[i], medias[j], covarianzas[j])
        gamma[i] /= np.sum(gamma[i])  # Normalizar
    return gamma

def maximization_step(X, gamma):
    """Actualiza los parámetros del modelo usando gamma."""
    n, d = X.shape
    k = gamma.shape[1]
    
    Nk = np.sum(gamma, axis=0)
    medias = np.array([np.sum(gamma[:, j][:, np.newaxis] * X, axis=0) / Nk[j] for j in range(k)])
    pesos = Nk / n
    
    covarianzas = []
    for j in range(k):
        diff = X - medias[j]
        cov_j = np.dot((gamma[:, j][:, np.newaxis] * diff).T, diff) / Nk[j]
        cov_j += 1e-6 * np.eye(d)  # Estabilización
        covarianzas.append(cov_j)

    return medias, covarianzas, pesos

def log_likelihood(X, medias, covarianzas, pesos):
    """Calcula el log-likelihood total para evaluar convergencia."""
    n = X.shape[0]
    k = len(medias)
    ll = 0
    for i in range(n):
        temp = 0
        for j in range(k):
            temp += pesos[j] * pdf_multivariada(X[i], medias[j], covarianzas[j])
        ll += np.log(temp + 1e-10)
    return ll

def gmm(X, k, max_iter=100, tol=1e-4):
    """Algoritmo EM para GMM."""
    medias, covarianzas, pesos = inicializar_gmm(X, k)
    log_likelihood_old = -np.inf

    for _ in range(max_iter):
        gamma = expectation_step(X, medias, covarianzas, pesos)
        medias, covarianzas, pesos = maximization_step(X, gamma)
        ll_new = log_likelihood(X, medias, covarianzas, pesos)

        if abs(ll_new - log_likelihood_old) < tol:
            break
        log_likelihood_old = ll_new

    labels = np.argmax(gamma, axis=1)
    return medias, covarianzas, pesos, labels