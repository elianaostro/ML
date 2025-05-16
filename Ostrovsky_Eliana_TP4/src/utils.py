import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def cargar_datos(ruta_csv: str) -> np.ndarray:
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(ruta_csv).values

def graficar_clusters(X: np.ndarray, labels: np.ndarray, centroides: np.ndarray):
    """Grafica los puntos de datos coloreados por cluster y sus centroides."""
    k = len(centroides)
    plt.figure(figsize=(8, 6))
    for i in range(k):
        puntos_cluster = X[labels == i]
        plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], label=f'Cluster {i}')
    plt.scatter(centroides[:, 0], centroides[:, 1], color='black', marker='x', s=100, label='Centroides')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'Clusters encontrados por K-means (K={k})')
    # plt.legend()
    plt.grid(True)
    plt.show()

def graficar_dbscan(X, etiquetas):
    """Grafica los clusters de DBSCAN, incluyendo ruido como puntos negros."""
    unique_labels = np.unique(etiquetas)
    colores = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        puntos = X[etiquetas == label]
        if label == -1:
            plt.scatter(puntos[:, 0], puntos[:, 1], c='black', label='Ruido')
        else:
            plt.scatter(puntos[:, 0], puntos[:, 1], color=colores[i], label=f'Cluster {label}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('DBSCAN desde cero')
    plt.legend()
    plt.grid(True)
    plt.show()

def graficar_metodo_del_codo(inercias: dict, k_max: int, k_min:int = 2):
    """Grafica el método del codo."""
    plt.figure(figsize=(8, 5))
    inercia = [d['inercia'] for d in inercias]
    plt.plot(range(k_min, k_max + 1), inercia , 'o-')
    plt.xlabel('Número de clusters K')
    plt.ylabel('Suma de distancias (L)')
    plt.title('Método del codo')
    plt.grid(True)
    plt.show()
