import numpy as np
import matplotlib.pyplot as plt

def center_data(X):
    """Centra los datos restando la media de cada columna."""
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    return X_centered, mean

def compute_covariance_matrix(X_centered):
    """Calcula la matriz de covarianza."""
    return np.cov(X_centered, rowvar=False)

def compute_pca(cov_matrix):
    """Devuelve autovalores y autovectores ordenados de mayor a menor."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx_sorted = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx_sorted], eigenvectors[:, idx_sorted]

def project_data(X_centered, eigenvectors, k):
    """Proyecta los datos en los k autovectores principales."""
    W_k = eigenvectors[:, :k]
    return X_centered @ W_k

def reconstruct_data(X_projected, eigenvectors, k, mean):
    """Reconstruye los datos a partir de la proyección PCA."""
    W_k = eigenvectors[:, :k]
    return X_projected @ W_k.T + mean

def reconstruction_error(X_original, X_reconstructed):
    """Calcula el error cuadrático medio de reconstrucción."""
    return np.mean((X_original - X_reconstructed) ** 2)

def pca_reconstruction_errors(X, X_centered, eigenvectors, mean, ks):
    """
    Calcula el error de reconstrucción para múltiples k, usando proyección parcial.
    Asume que X ya está centrado y que eigenvectors están ordenados.
    """
    max_k = max(ks)
    W_all = eigenvectors[:, :max_k]
    X_proj_all = X_centered @ W_all

    errors = []
    for k in ks:
        X_proj_k = X_proj_all[:, :k]
        W_k = W_all[:, :k]
        X_rec = X_proj_k @ W_k.T + mean
        err = np.mean((X - X_rec) ** 2)
        errors.append(err)

    return errors
