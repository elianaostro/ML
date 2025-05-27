import numpy as np
from src.clustering_common import inicializar_centroides, pdf_gaussiana
from src.kmeans import kmeans

def inicializar_gmm(X, k):
    medias, labels, _ = kmeans(X, k, seed = None)
    n, d = X.shape
    pi = np.array([np.mean(labels == j) for j in range(k)])
    covs = []
    for j in range(k):
        Xj = X[labels == j]
        if len(Xj) > 1:
            cov = np.cov(Xj.T) + 1e-6 * np.eye(d)
        else:
            cov = np.eye(d)
        covs.append(cov)
    return medias, covs, pi

def expectation(X, medias, covs, pi):
    n, k = X.shape[0], len(medias)
    gamma = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            gamma[i, j] = pi[j] * pdf_gaussiana(X[i], medias[j], covs[j])
        gamma[i] /= np.sum(gamma[i]) + 1e-10
    return gamma

def maximization(X, gamma):
    n, d = X.shape
    k = gamma.shape[1]
    Nk = gamma.sum(axis=0)
    pi = Nk / n
    medias = np.array([np.sum(gamma[:, j][:, None] * X, axis=0) / Nk[j] for j in range(k)])
    covs = []
    for j in range(k):
        diff = X - medias[j]
        cov = (gamma[:, j][:, None] * diff).T @ diff / Nk[j]
        cov += 1e-6 * np.eye(d)
        covs.append(cov)
    return medias, covs, pi

def log_likelihood(X, medias, covs, pi):
    ll = 0
    for i in range(X.shape[0]):
        s = sum(pi[j] * pdf_gaussiana(X[i], medias[j], covs[j]) for j in range(len(medias)))
        ll += np.log(s + 1e-10)
    return ll

def gmm(X, k, max_iter=100, tol=1e-4):
    medias, covs, pi = inicializar_gmm(X, k)
    prev_ll = -np.inf
    for _ in range(max_iter):
        gamma = expectation(X, medias, covs, pi)
        medias, covs, pi = maximization(X, gamma)
        ll = log_likelihood(X, medias, covs, pi)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    labels = np.argmax(gamma, axis=1)
    return medias, covs, pi, labels
