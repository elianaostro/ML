import numpy as np
from src.clustering_common import inicializar_centroides, pdf_gaussiana
from src.kmeans import iter_kmeans
from src.utils import update_cluster_progress_bar

def inicializar_gmm(X, k):
    medias, labels, _ = iter_kmeans(X, k)
    n, d = X.shape
    pi = np.bincount(labels, minlength=k) / n
    covs = []
    for j in range(k):
        Xj = X[labels == j]
        if Xj.shape[0] > 1:
            cov = np.cov(Xj, rowvar=False) + 1e-6 * np.eye(d)
        else:
            cov = np.diag(np.var(X, axis=0)) * 0.1 + 1e-6 * np.eye(d)
        covs.append(cov)

    return medias, covs, pi

def pdfs_gaussianas(X, medias, covs):
    n, d = X.shape
    k = medias.shape[0]
    pdfs = np.zeros((n, k))

    for j in range(k):
        diff = X - medias[j]  # (n, d)
        inv = np.linalg.inv(covs[j])
        det = np.linalg.det(covs[j])
        norm = 1 / np.sqrt((2 * np.pi) ** d * det)
        exp = np.einsum('ni,ij,nj->n', diff, inv, diff)  # vectorizado: (n,)
        pdfs[:, j] = norm * np.exp(-0.5 * exp)

    return pdfs

def expectation(X, n, k, medias, covs, pi):
    pdf_vals = pdfs_gaussianas(X, medias, covs)  # (n, k)
    gamma = pdf_vals * pi  # Broadcasting: (n, k)
    gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10
    return gamma

def maximization(X, n, d, k, gamma, reg=1e-3):
    Nk = gamma.sum(axis=0)
    pi = Nk / n
    medias = np.array([np.sum(gamma[:, j][:, None] * X, axis=0) / Nk[j] for j in range(k)])
    covs = []
    for j in range(k):
        diff = X - medias[j]
        cov = (gamma[:, j][:, None] * diff).T @ diff / Nk[j]
        cov += reg * np.eye(d)  # regularización aquí
        covs.append(cov)
    return medias, covs, pi

def log_likelihood(X, medias, covs, pi):
    pdf_vals = pdfs_gaussianas(X, medias, covs)  # (n, k)
    weighted_sum = np.dot(pdf_vals, pi)  # (n,)
    return np.sum(np.log(weighted_sum + 1e-10))

def gmm(X, k, max_iter=100, tol=1e-4):
    medias, covs, pi = inicializar_gmm(X, k)
    print(f"Inicializando GMM con {k} clusters...")
    n, d = X.shape
    prev_ll = -np.inf
    for i in range(max_iter):
        gamma = expectation(X, n, k, medias, covs, pi)
        medias, covs, pi = maximization(X, n, d, k, gamma)
        ll = log_likelihood(X, medias, covs, pi)
        update_cluster_progress_bar(i + 1, max_iter, metrics={'loglik': ll})
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    labels = np.argmax(gamma, axis=1)
    return medias, covs, pi, labels