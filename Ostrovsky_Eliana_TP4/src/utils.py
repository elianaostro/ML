import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def cargar_datos(ruta):
    """Carga datos desde un archivo CSV y los devuelve como un array NumPy."""
    return pd.read_csv(ruta).values

def graficar_clusters(X, etiquetas, centroides=None, titulo="Clusters"):
    """
    Grafica los puntos del dataset coloreados según su etiqueta.
    Si se especifican centroides, los marca como cruces negras.
    Devuelve la figura para poder hacer un layout.
    """
    etiquetas_unicas = np.unique(etiquetas)
    colores = plt.cm.tab10(np.linspace(0, 1, len(etiquetas_unicas)))

    fig = plt.figure(figsize=(8, 6))

    for i, etiqueta in enumerate(etiquetas_unicas):
        puntos = X[etiquetas == etiqueta]
        if etiqueta == -1:
            plt.scatter(puntos[:, 0], puntos[:, 1], c='black', label="Ruido", s=30)
        else:
            plt.scatter(puntos[:, 0], puntos[:, 1], color=colores[i], label=f"Cluster {etiqueta}", s=30)

    if centroides is not None:
        plt.scatter(centroides[:, 0], centroides[:, 1], color='black', marker='x', s=100, label='Centroides')

    plt.title(titulo)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True)
    
    return fig

def escalar_estandar(X):
    """
    Aplica normalización Z-score manual (media 0, varianza 1).
    """
    media = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - media) / std

def escalar_minmax(X):
    """
    Aplica normalización Min-Max (escala entre 0 y 1).
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)

def update_cluster_progress_bar(current_iter, total_iters, bar_length=50, metrics=None):
    """
    Muestra una barra de progreso para las iteraciones de clustering, con métricas opcionales.

    Args:
        current_iter: Iteración actual (comenzando en 1)
        total_iters: Número total de iteraciones
        bar_length: Largo de la barra de progreso
        metrics: Diccionario de métricas a mostrar (opcional)
    """
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