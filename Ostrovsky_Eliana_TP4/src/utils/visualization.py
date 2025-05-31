import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from typing import Tuple, Optional, List


def plot_clusters(X, labels, centroids=None, title="Clusters"):
    """
    Dibuja los puntos de datos coloreados por etiqueta, y los centroides si existen.
    """
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        if label == -1:
            color = 'k'
            marker = 'x'
            label_name = 'Ruido'
        else:
            color = colors(i)
            marker = 'o'
            label_name = f'Cluster {label}'
        
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], c=[color], label=label_name, marker=marker, alpha=0.7)

    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='*', label='Centroides')

    plt.title(title)
    if len(unique_labels) > 1:
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.grid(True)
    plt.show()


def plot_elbow(losses, k_range, title="Gráfico del Método del Codo"):
    """
    Dibuja el gráfico del método del codo.
    Parameters:
        - losses: lista de pérdidas (L o -log-likelihood)
        - k_range: lista de valores de K
    """
    fig = plt.figure(figsize=(8, 4))
    plt.plot(k_range, losses, marker='o')
    plt.xlabel('Cantidad de clusters (K)')
    plt.ylabel('inercia')
    plt.title(title)
    plt.grid(True)
    return fig


def plot_dbscan_scores(scores):
    """
    Dibuja gráficos comparativos a partir de los resultados de explore_dbscan_params.
    """
    df = pd.DataFrame(scores, columns=["eps", "min_samples", "silhouette"])
    pivot_table = df.pivot(index="min_samples", columns="eps", values="silhouette")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Silhouette score para combinaciones de DBSCAN")
    plt.xlabel("eps")
    plt.ylabel("min_samples")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(confusion_matrix: np.ndarray, figsize: Tuple[int, int] = (10, 8), 
                          class_subset: Optional[List[int]] = None) -> None:
    """
    Función para graficar la matriz de confusión
    """
    if class_subset is not None:
        cm = confusion_matrix[np.ix_(class_subset, class_subset)]
    else:
        cm = confusion_matrix
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    num_classes = cm.shape[0]
    tick_marks = np.arange(num_classes)
    if class_subset is not None:
        plt.xticks(tick_marks, class_subset)
        plt.yticks(tick_marks, class_subset)
    else:
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
