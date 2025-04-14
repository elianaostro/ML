# src/metrics.py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Calcula la matriz de confusión.

    Args:
        y_true (array-like): Etiquetas verdaderas.
        y_pred (array-like): Etiquetas predichas.
        labels (array-like, optional): Lista de etiquetas para indexar la matriz.
                                      Si es None, se usan las etiquetas presentes en y_true y y_pred.

    Returns:
        np.ndarray: Matriz de confusión donde CM[i, j] es el número de observaciones
                    conocidas en la clase i pero predichas en la clase j.
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    
    n_labels = len(labels)
    label_to_ind = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        true_label_ind = label_to_ind.get(true)
        pred_label_ind = label_to_ind.get(pred)
        
        # Solo incrementar si ambas etiquetas están en la lista de labels
        if true_label_ind is not None and pred_label_ind is not None:
            cm[true_label_ind, pred_label_ind] += 1
            
    return cm

def accuracy_score(y_true, y_pred):
    """Calcula la precisión global (accuracy)"""
    return np.mean(np.array(y_true) == np.array(y_pred))

def precision_score(y_true, y_pred, labels=None, average='binary', zero_division=0):
    """
    Calcula la precisión.

    Args:
        y_true (array-like): Etiquetas verdaderas.
        y_pred (array-like): Etiquetas predichas.
        labels (array-like, optional): El conjunto de etiquetas a incluir.
        average (str, optional): ['binary', 'micro', 'macro', 'weighted', None]
            - 'binary': Solo reporta para la clase especificada por `pos_label`. Asume clase 1 si no se especifica.
            - 'micro': Calcula métricas globalmente contando TP, FN, FP totales.
            - 'macro': Calcula métricas para cada etiqueta y encuentra su media no ponderada.
            - 'weighted': Calcula métricas para cada etiqueta y encuentra su media ponderada por soporte.
            - None: Devuelve las puntuaciones para cada clase.
        zero_division (int or float): Valor a retornar cuando hay división por cero.

    Returns:
        float or np.ndarray: Puntuación de precisión o array de puntuaciones por clase.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    n_labels = len(labels)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    support = cm.sum(axis=1) # Número de instancias verdaderas por clase

    if average == 'binary':
        if n_labels != 2:
            raise ValueError("average='binary' is only supported for binary classification")
        # Asumimos que la clase positiva es la segunda etiqueta encontrada si no se especifica
        pos_label_idx = 1 
        tp_binary = tp[pos_label_idx]
        fp_binary = fp[pos_label_idx]
        precision = tp_binary / (tp_binary + fp_binary) if (tp_binary + fp_binary) > 0 else zero_division
        return precision
        
    elif average == 'micro':
        tp_total = np.sum(tp)
        fp_total = np.sum(fp)
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else zero_division
        return precision
        
    else: # macro, weighted, None
        precision_per_class = np.zeros(n_labels)
        for i in range(n_labels):
            denominator = tp[i] + fp[i]
            precision_per_class[i] = tp[i] / denominator if denominator > 0 else zero_division
            
        if average is None or average == 'none':
            return precision_per_class
        elif average == 'macro':
            return np.mean(precision_per_class)
        elif average == 'weighted':
            if np.sum(support) == 0:
                 return zero_division
            return np.average(precision_per_class, weights=support)
        else:
            raise ValueError("average parameter must be 'binary', 'micro', 'macro', 'weighted', or None")


def recall_score(y_true, y_pred, labels=None, average='binary', zero_division=0):
    """
    Calcula el recall (sensibilidad).

    Args:
        y_true (array-like): Etiquetas verdaderas.
        y_pred (array-like): Etiquetas predichas.
        labels (array-like, optional): El conjunto de etiquetas a incluir.
        average (str, optional): ['binary', 'micro', 'macro', 'weighted', None]
            (Ver docstring de precision_score para detalles)
        zero_division (int or float): Valor a retornar cuando hay división por cero.

    Returns:
        float or np.ndarray: Puntuación de recall o array de puntuaciones por clase.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    n_labels = len(labels)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    if average == 'binary':
        if n_labels != 2:
            raise ValueError("average='binary' is only supported for binary classification")
        pos_label_idx = 1
        tp_binary = tp[pos_label_idx]
        fn_binary = fn[pos_label_idx]
        recall = tp_binary / (tp_binary + fn_binary) if (tp_binary + fn_binary) > 0 else zero_division
        return recall
        
    elif average == 'micro':
        tp_total = np.sum(tp)
        fn_total = np.sum(fn) # En micro, FN total es igual a FP total
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else zero_division
        return recall # Note: Micro-precision == Micro-recall == Micro-F1 == Accuracy
        
    else: # macro, weighted, None
        recall_per_class = np.zeros(n_labels)
        for i in range(n_labels):
            denominator = tp[i] + fn[i] # = support[i]
            recall_per_class[i] = tp[i] / denominator if denominator > 0 else zero_division
            
        if average is None or average == 'none':
            return recall_per_class
        elif average == 'macro':
            return np.mean(recall_per_class)
        elif average == 'weighted':
            if np.sum(support) == 0:
                return zero_division
            return np.average(recall_per_class, weights=support)
        else:
            raise ValueError("average parameter must be 'binary', 'micro', 'macro', 'weighted', or None")


def f1_score(y_true, y_pred, labels=None, average='binary', zero_division=0):
    """
    Calcula el F1-score.

    Args:
        y_true (array-like): Etiquetas verdaderas.
        y_pred (array-like): Etiquetas predichas.
        labels (array-like, optional): El conjunto de etiquetas a incluir.
        average (str, optional): ['binary', 'micro', 'macro', 'weighted', None]
            (Ver docstring de precision_score para detalles)
        zero_division (int or float): Valor a retornar cuando hay división por cero.

    Returns:
        float or np.ndarray: Puntuación F1 o array de puntuaciones por clase.
    """
    precision = precision_score(y_true, y_pred, labels=labels, average=average, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, labels=labels, average=average, zero_division=zero_division)

    if average == 'binary' or average == 'micro' or average == 'macro' or average == 'weighted':
        # Si precision y recall son escalares
        denominator = precision + recall
        f1 = (2 * precision * recall) / denominator if denominator > 0 else zero_division
        return f1
    elif average is None or average == 'none':
        # Si precision y recall son arrays
        f1 = np.zeros_like(precision)
        valid = (precision + recall) > 0
        f1[valid] = (2 * precision[valid] * recall[valid]) / (precision[valid] + recall[valid])
        f1[~valid] = zero_division
        return f1
    else:
        raise ValueError("average parameter must be 'binary', 'micro', 'macro', 'weighted', or None")


# --- Funciones ROC y PR (principalmente para binario o OvR) ---

def roc_curve(y_true, y_proba):
    """
    Calcula la curva ROC (Receiver Operating Characteristic).
    Diseñado principalmente para clasificación binaria o el enfoque One-vs-Rest (OvR)
    para la clase positiva en multiclase.

    Args:
        y_true (array-like): Etiquetas binarias verdaderas.
        y_proba (array-like): Probabilidades de la clase positiva (o puntajes de confianza).

    Returns:
        tuple: (fpr, tpr, thresholds)
    """
    if y_proba.ndim > 1 and y_proba.shape[1] >= 2:
         # Asume que la segunda columna es la probabilidad de la clase positiva
        y_scores = y_proba[:, 1]
    else:
        y_scores = y_proba # Si ya es 1D

    # Ordenar por scores descendentes
    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = np.array(y_true)[desc_score_indices] # Asegurarse que es numpy array

    # Contadores para TP y FP
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    thresholds = y_scores[threshold_idxs]

    # Calcular TPs y FPs acumulados
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps # Número de negativos hasta ese índice - TN = FP

    # Obtener número total de positivos y negativos
    # Manejar caso donde solo hay una clase en y_true
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    if len(unique_true) == 1:
        if unique_true[0] == 1: # Solo positivos
             n_positives = counts_true[0]
             n_negatives = 0
        else: # Solo negativos
             n_positives = 0
             n_negatives = counts_true[0]
    else:
        # Asume que 1 es la clase positiva
        pos_idx = np.where(unique_true == 1)[0][0]
        neg_idx = 1 - pos_idx
        n_positives = counts_true[pos_idx]
        n_negatives = counts_true[neg_idx]


    # Calcular TPR y FPR
    tpr = tps / n_positives if n_positives > 0 else np.zeros_like(tps)
    fpr = fps / n_negatives if n_negatives > 0 else np.zeros_like(fps)

    # Añadir punto (0, 0) inicial
    fpr = np.r_[0, fpr]
    tpr = np.r_[0, tpr]
    thresholds = np.r_[thresholds[0] + 1, thresholds] # Añadir un umbral > max(score)

    return fpr, tpr, thresholds


def pr_curve(y_true, y_proba):
    """
    Calcula la curva Precision-Recall.
    Diseñado principalmente para clasificación binaria o el enfoque OvR.

    Args:
        y_true (array-like): Etiquetas binarias verdaderas.
        y_proba (array-like): Probabilidades de la clase positiva.

    Returns:
        tuple: (precision, recall, thresholds)
    """
    if y_proba.ndim > 1 and y_proba.shape[1] >= 2:
        y_scores = y_proba[:, 1]
    else:
        y_scores = y_proba

    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = np.array(y_true)[desc_score_indices] # Asegurarse que es numpy array

    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    thresholds = y_scores[threshold_idxs]

    tps = np.cumsum(y_true)[threshold_idxs]
    # fps = 1 + threshold_idxs - tps # No se usa directamente en PR
    pred_positives = 1 + threshold_idxs # Número total de predichos positivos en cada umbral

    # Obtener número total de positivos reales
    n_positives = np.sum(y_true == 1)

    precision = tps / pred_positives if n_positives > 0 else np.zeros_like(tps)
    recall = tps / n_positives if n_positives > 0 else np.zeros_like(tps)

    # Añadir punto (recall=0, precision=1) o un punto razonable inicial
    # Scikit-learn añade (0, 1) - ajustando para que tenga sentido con los umbrales
    precision = np.r_[precision[0], precision] # Podría ser 1 si el primer punto es perfecto
    recall = np.r_[0, recall]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    # Corrección para que el último punto sea (1, P(clase positiva))
    # Si el recall es 1, la precisión es la proporción de positivos totales
    # last_precision = n_positives / len(y_true) if len(y_true) > 0 else 0
    # precision = np.r_[precision, last_precision]
    # recall = np.r_[recall, 1.0]
    
    # Devolver ordenados por recall ascendente (como es usual)
    # No es estrictamente necesario aquí ya que los umbrales van descendiendo
    
    return precision, recall, thresholds


def auc(x, y):
    """
    Calcula el área bajo la curva (AUC) usando la regla del trapecio.
    Asume que los puntos x están ordenados.

    Args:
        x (array-like): Valores del eje x (e.g., FPR or Recall).
        y (array-like): Valores del eje y (e.g., TPR or Precision).

    Returns:
        float: Área bajo la curva.
    """
    # Asegurarse que están ordenados por x para trapecio
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Calcular AUC usando la regla del trapecio
    area = np.trapz(y_sorted, x_sorted)
    return area


def plot_confusion_matrix(y_true, y_pred, labels=None, display_labels=None, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Grafica la matriz de confusión.

    Args:
        y_true (array-like): Etiquetas verdaderas.
        y_pred (array-like): Etiquetas predichas.
        labels (array-like, optional): Lista de etiquetas para ordenar la matriz. Si None, usa np.unique.
        display_labels (array-like, optional): Nombres para mostrar en los ejes. Si None, usa `labels`.
        title (str): Título del gráfico.
        cmap: Colormap a usar.
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    if display_labels is None:
        display_labels = labels

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=display_labels, yticklabels=display_labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotar etiquetas del eje x si son muchas o largas
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Añadir texto con los valores
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.show()


def classification_report(y_true, y_pred, labels=None, target_names=None, zero_division=0):
    """
    Genera un reporte de clasificación con las principales métricas por clase.

    Args:
        y_true (array-like): Etiquetas verdaderas.
        y_pred (array-like): Etiquetas predichas.
        labels (array-like, optional): Lista de etiquetas a incluir en el reporte.
                                      Si None, se usan todas las etiquetas presentes.
        target_names (list of str, optional): Nombres de las clases para mostrar.
                                             Debe corresponder con `labels`.
        zero_division (int or float): Valor para métricas cuando hay división por cero.

    Returns:
        dict: Diccionario con las métricas por clase y agregadas.
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
        labels.sort()

    if target_names is None:
        target_names = [str(l) for l in labels]
    elif len(labels) != len(target_names):
         raise ValueError("Longitud de labels y target_names no coincide.")

    # Calcular métricas por clase
    p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=zero_division)
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=zero_division)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=zero_division)
    
    # Calcular soporte (instancias verdaderas por clase)
    true_counts = Counter(y_true)
    support = np.array([true_counts.get(l, 0) for l in labels])

    # Calcular métricas agregadas
    accuracy = accuracy_score(y_true, y_pred)
    macro_p = np.average(p)
    macro_r = np.average(r)
    macro_f1 = np.average(f1)
    weighted_p = np.average(p, weights=support)
    weighted_r = np.average(r, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    # --- Formatear el reporte ---
    headers = ["precision", "recall", "f1-score", "support"]
    # Determinar ancho máximo de nombres para alinear
    max_name_width = max(len(name) for name in target_names)
    width = max(max_name_width, len("weighted avg"))
    head_fmt = '{:>{width}s} ' + ' {:>9s}' * len(headers)
    report = head_fmt.format('', *headers, width=width) + '\n\n'

    row_fmt = '{:>{width}s} ' + ' {:>9.2f}' * 3 + ' {:>9d}\n'
    for i, label in enumerate(labels):
        report += row_fmt.format(target_names[i], p[i], r[i], f1[i], support[i], width=width)

    report += '\n'

    # --- Resumen de promedios ---
    avg_row_fmt = '{:>{width}s} ' + ' {:>9s}' * 2 + ' {:>9.2f} {:>9d}\n'
    report += avg_row_fmt.format('accuracy', '', '', accuracy, np.sum(support), width=width)
    
    avg_row_fmt = '{:>{width}s} ' + ' {:>9.2f}' * 3 + ' {:>9d}\n'
    report += avg_row_fmt.format('macro avg', macro_p, macro_r, macro_f1, np.sum(support), width=width)
    report += avg_row_fmt.format('weighted avg', weighted_p, weighted_r, weighted_f1, np.sum(support), width=width)

    print(report)

    # Crear el diccionario con las métricas
    metrics_dict = {
        "classes": {
            target_names[i]: {
                "precision": p[i],
                "recall": r[i],
                "f1-score": f1[i],
                "support": support[i]
            } for i in range(len(labels))
        },
        "accuracy": accuracy,
        "macro avg": {
            "precision": macro_p,
            "recall": macro_r,
            "f1-score": macro_f1,
            "support": np.sum(support)
        },
        "weighted avg": {
            "precision": weighted_p,
            "recall": weighted_r,
            "f1-score": weighted_f1,
            "support": np.sum(support)
        }
    }

    return metrics_dict


def display_binary_metrics(y_true, y_pred, y_proba, pos_label=1, labels=None, target_names=None):
    """
    Calcula y muestra las métricas de performance para clasificación BINARIA.
    Incluye reporte por clase, matriz de confusión y curvas ROC/PR.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        y_proba: Probabilidades predichas (array Nx2 o N).
        pos_label: La etiqueta considerada como clase positiva.
        labels (list, optional): Orden de las etiquetas.
        target_names (list, optional): Nombres de las clases.
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
        labels.sort()
        if len(labels) != 2:
             print("Advertencia: display_binary_metrics espera 2 clases. Se encontraron:", labels)
             # Podría lanzar error o continuar mostrando para la pos_label indicada
             
    if target_names is None:
        target_names = [str(l) for l in labels]

    print("\n--- Reporte de Clasificación ---")
    classification_report(y_true, y_pred, labels=labels, target_names=target_names)

    print("\n--- Matriz de Confusión ---")
    plot_confusion_matrix(y_true, y_pred, labels=labels, display_labels=target_names)

    # Asegurarse que y_proba tiene las probabilidades de la clase positiva
    if y_proba.ndim > 1 and y_proba.shape[1] >= 2:
        # Encontrar el índice de la clase positiva
        try:
             pos_label_idx = list(labels).index(pos_label)
             y_scores = y_proba[:, pos_label_idx]
        except ValueError:
             print(f"Advertencia: pos_label '{pos_label}' no encontrada en labels '{labels}'. Usando columna 1 para ROC/PR.")
             y_scores = y_proba[:, 1] #Fallback a la segunda columna
    else:
        y_scores = y_proba # Asumir que ya es la prob de la clase positiva

    print("\n--- Curvas de Performance ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Curva ROC y AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('Tasa de Falsos Positivos (FPR)')
    axes[0].set_ylabel('Tasa de Verdaderos Positivos (TPR)')
    axes[0].set_title('Curva ROC')
    axes[0].legend(loc="lower right")
    axes[0].grid(True)

    # Curva Precision-Recall y AUC-PR
    precision, recall, pr_thresholds = pr_curve(y_true, y_scores)
    pr_auc = auc(recall, precision) # AUC PR usa recall en el eje x
    axes[1].plot(recall, precision, color='blue', lw=2, label=f'Curva PR (AUC = {pr_auc:.2f})')
    # Línea base: proporción de positivos
    n_positives = np.sum(np.array(y_true) == pos_label)
    baseline = n_positives / len(y_true) if len(y_true) > 0 else 0
    axes[1].axhline(baseline, color='grey', lw=1, linestyle='--', label=f'Baseline ({baseline:.2f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Curva Precision-Recall')
    axes[1].legend(loc="lower left")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def display_multiclass_metrics(y_true, y_pred, y_proba=None, labels=None, target_names=None, title_suffix=""):
    """
    Calcula y muestra las métricas de performance clave para clasificación MULTICLASE.
    Muestra el reporte de clasificación, la matriz de confusión y, si se proveen
    probabilidades, las curvas ROC y PR bajo la estrategia One-vs-Rest (OvR).

    Args:
        y_true (array-like): Etiquetas verdaderas.
        y_pred (array-like): Etiquetas predichas.
        y_proba (array-like, optional): Probabilidades predichas por clase (n_samples, n_classes).
                                        Necesario para graficar curvas ROC y PR.
        labels (list, optional): Lista de etiquetas a incluir y ordenar. Si None, usa las presentes.
        target_names (list, optional): Nombres para mostrar para las clases. Si None, usa los labels.
        title_suffix (str, optional): Sufijo para añadir a los títulos de los gráficos/reportes.
    """
    y_true = np.array(y_true) # Asegurar que es numpy array

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
        labels.sort()
    n_classes = len(labels)
    label_to_ind = {label: i for i, label in enumerate(labels)}

    if target_names is None:
        target_names = [str(l) for l in labels]
    elif len(labels) != len(target_names):
         raise ValueError("Longitud de labels y target_names no coincide.")

    report_title = f"--- Reporte de Clasificación Multiclase {title_suffix} ---"
    cm_title = f"Matriz de Confusión {title_suffix}"
    roc_title = f"Curvas ROC (One-vs-Rest) {title_suffix}"
    pr_title = f"Curvas Precision-Recall (One-vs-Rest) {title_suffix}"

    print("\n" + report_title)
    classification_report(y_true, y_pred, labels=labels, target_names=target_names)

    print(f"\n--- {cm_title} ---")
    plot_confusion_matrix(y_true, y_pred, labels=labels, display_labels=target_names, title=cm_title)

    # --- Gráficos ROC y PR (si se proporcionan probabilidades) ---
    if y_proba is not None:
        if y_proba.shape[1] != n_classes:
             raise ValueError(f"La forma de y_proba ({y_proba.shape}) no coincide con el número de clases ({n_classes}) según 'labels'")

        print(f"\n--- Curvas de Performance (One-vs-Rest) {title_suffix} ---")
        
        # Configurar plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Ajustar tamaño según sea necesario
        
        # Colores para las curvas (ciclo si hay más clases que colores)
        colors = plt.cm.get_cmap('tab10', n_classes) 

        # --- Curva ROC (OvR) ---
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Azar')
        print("AUC ROC (OvR):")
        for i, label in enumerate(labels):
            class_index = label_to_ind[label]
            y_true_binary = (y_true == label).astype(int)
            y_scores = y_proba[:, class_index]

            # Asegurarse que hay ambas clases en y_true_binary para calcular la curva
            if len(np.unique(y_true_binary)) < 2:
                 print(f"  - Clase '{target_names[i]}': No se puede calcular ROC (solo una clase presente en y_true)")
                 continue # Saltar esta clase para ROC/PR

            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            roc_auc = auc(fpr, tpr)
            print(f"  - Clase '{target_names[i]}': {roc_auc:.3f}")
            axes[0].plot(fpr, tpr, color=colors(i), lw=2, 
                         label=f'Clase {target_names[i]} (AUC = {roc_auc:.2f})')

        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('Tasa de Falsos Positivos (FPR)')
        axes[0].set_ylabel('Tasa de Verdaderos Positivos (TPR)')
        axes[0].set_title(roc_title)
        axes[0].legend(loc="lower right", fontsize='small')
        axes[0].grid(True)


        # --- Curva Precision-Recall (OvR) ---
        print("\nAUC PR (OvR):")
        for i, label in enumerate(labels):
            class_index = label_to_ind[label]
            y_true_binary = (y_true == label).astype(int)
            y_scores = y_proba[:, class_index]

            if len(np.unique(y_true_binary)) < 2:
                 print(f"  - Clase '{target_names[i]}': No se puede calcular PR (solo una clase presente en y_true)")
                 continue # Saltar esta clase para ROC/PR

            precision, recall, _ = pr_curve(y_true_binary, y_scores)
            # Nota: El AUC PR puede ser menos informativo si las clases están muy desbalanceadas
            pr_auc = auc(recall, precision) # AUC PR usa recall en eje x
            print(f"  - Clase '{target_names[i]}': {pr_auc:.3f}")
            
            # Graficar línea base (prevalencia de la clase)
            baseline = np.sum(y_true_binary) / len(y_true_binary) if len(y_true_binary) > 0 else 0
            axes[1].plot(recall, precision, color=colors(i), lw=2, 
                         label=f'Clase {target_names[i]} (AUC = {pr_auc:.2f})')
            axes[1].plot([0, 1], [baseline, baseline], color=colors(i), lw=1, linestyle='--', 
                         label=f'_Baseline Clase {target_names[i]} ({baseline:.2f})') # '_' evita que aparezca en la leyenda principal

        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(pr_title)
        # Añadir una leyenda separada para las baselines podría ser útil si hay muchas clases
        axes[1].legend(loc="lower left", fontsize='small') 
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()

    else:
        print("\nAdvertencia: No se proporcionaron probabilidades (y_proba). No se graficarán curvas ROC ni PR.")
