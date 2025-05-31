import matplotlib.pyplot as plt
import numpy as np
from src.PCA import project_data, reconstruct_data

def show_pca_reconstructions(X, X_centered, mean, eigenvectors, k_values, y, num_classes=10):
    """
    Muestra una grilla de imágenes: original y reconstrucciones para cada k.
    Muestra una imagen por cada clase/etiqueta.
    """
    sorted_items = sorted(k_values.items(), reverse=True)  
    ks = [k for _, k in sorted_items]
    n_rows = 1 + len(ks)  
    
    unique_labels = np.unique(y)[:num_classes]
    num_images = len(unique_labels)
    class_indices = [np.where(y == label)[0][0] for label in unique_labels]
    
    fig, axs = plt.subplots(n_rows, num_images, figsize=(num_images, n_rows + 1))
    axs = np.atleast_2d(axs)

    for i, idx in enumerate(class_indices):
        axs[0, i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[0, i].set_title(f"Label: {y[idx]}", fontsize=10)
    
    axs[0, 0].set_ylabel("Original", fontsize=12)

    for row, (explain_var, k) in enumerate(sorted_items, start=1):
        X_proj = project_data(X_centered, eigenvectors, k)
        X_rec = reconstruct_data(X_proj, eigenvectors, k, mean)
        for i, idx in enumerate(class_indices):
            axs[row, i].imshow(X_rec[idx].reshape(28, 28), cmap='gray')
            axs[row, i].set_xticks([])
            axs[row, i].set_yticks([])
        axs[row, 0].set_ylabel(f"k={k}\n({explain_var:.2f})", fontsize=11)

    plt.suptitle("Reconstrucciones con PCA para diferentes valores de k", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_pca_2d(X_centered, eigenvectors, y, title="PCA 2D"):
    """
    Proyecta el dataset centrado a 2D con PCA y grafica los puntos coloreados por su etiqueta.

    - X_centered: datos centrados (n_samples, n_features)
    - eigenvectors: autovectores ordenados (n_features, n_features)
    - y: etiquetas (n_samples,)
    """
    W_2d = eigenvectors[:, :2]
    X_2d = X_centered @ W_2d

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=10, alpha=0.7)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title(title)
    plt.colorbar(scatter, ticks=range(len(set(y))), label="Etiqueta")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pca_2d_per_class(X_centered, eigenvectors, y, title="PCA 2D por clase", max_classes=10):
    """
    Genera un subplot 2D por clase con la proyección PCA a 2 dimensiones.
    
    - X_centered: datos centrados
    - eigenvectors: autovectores ordenados
    - y: etiquetas (n_samples,)
    - max_classes: máximo número de clases a graficar
    """
    W_2d = eigenvectors[:, :2]
    X_2d = X_centered @ W_2d

    labels = np.unique(y)
    n_classes = min(len(labels), max_classes)
    n_cols = 5
    n_rows = int(np.ceil(n_classes / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    axs = axs.flatten()

    for i, label in enumerate(labels[:n_classes]):
        mask = y == label
        axs[i].scatter(X_2d[mask, 0], X_2d[mask, 1], s=10, alpha=0.7, color="tab:blue")
        axs[i].set_title(f"Clase {label}")
        axs[i].set_xlim(-2500, 2000)
        axs[i].set_ylim(-1500, 1500)
        axs[i].grid(True)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_pca_centroids_2d(X_centered, eigenvectors, y, title="Centroides en PCA 2D"):
    """
    Proyecta X a 2D con PCA y grafica un punto por clase (el centroide).
    
    - X_centered: datos centrados (n_samples, n_features)
    - eigenvectors: autovectores ordenados (n_features, n_features)
    - y: etiquetas (n_samples,)
    """

    labels = np.unique(y)
    centroids = []

    for label in labels:
        X_label = X_centered[y == label]
        centroid = X_label.mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)    
    
    W_2d = eigenvectors[:, :2]
    C_2d = centroids @ W_2d

    plt.figure(figsize=(8, 6))
    plt.scatter(C_2d[:, 0], C_2d[:, 1], c=labels, cmap='tab10', s=100, edgecolors='k')

    for i, label in enumerate(labels):
        plt.text(C_2d[i, 0] + 15, C_2d[i, 1] + 15, str(label), fontsize=10)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
