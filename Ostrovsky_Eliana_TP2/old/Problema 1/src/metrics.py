import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _get_confusion_counts(y_true, y_pred):
    """
    Calcula los valores de la matriz de confusión: TN, FP, FN, TP.

    Parameters:
    y_true (array-like): Etiquetas verdaderas (0 o 1).
    y_pred (array-like): Etiquetas predichas (0 o 1).

    Returns:
    tuple: TN, FP, FN, TP
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TN, FP, FN, TP

def calculate_confusion_matrix(y_true, y_pred):
    """
    Calcula la matriz de confusión 2x2.

    Returns:
    ndarray: Matriz [[TN, FP], [FN, TP]]
    """
    return np.array(_get_confusion_counts(y_true, y_pred)).reshape(2, 2)

def calculate_accuracy(y_true, y_pred):
    """
    Calcula la precisión global del modelo.

    Returns:
    float: Accuracy
    """
    return np.mean(y_true == y_pred)

def calculate_precision_recall_f1(y_true, y_pred):
    """
    Calcula precision, recall y F1-score para clasificación binaria.

    Returns:
    tuple: (precision, recall, f1_score)
    """
    _, FP, FN, TP = _get_confusion_counts(y_true, y_pred)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def _calculate_curve(y_true, y_proba, score_func, thresholds=100):
    """
    Calcula una curva de métricas (ej: ROC o Precision-Recall).

    Parameters:
    score_func (function): Función que recibe (y_true, y_pred) y devuelve (x, y) a graficar.
    
    Returns:
    tuple of np.ndarrays: Coordenadas x e y de la curva.
    """
    thresholds = np.linspace(0, 1, thresholds)
    scores_x, scores_y = [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        x, y = score_func(y_true, y_pred)
        scores_x.append(x)
        scores_y.append(y)

    return np.array(scores_x), np.array(scores_y)

def _tpr_fpr(y_true, y_pred):
    """
    Calcula la tasa de verdaderos positivos (TPR) y tasa de falsos positivos (FPR).

    Returns:
    tuple: (FPR, TPR)
    """
    TN, FP, FN, TP = _get_confusion_counts(y_true, y_pred)
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    return fpr, tpr

def _precision_recall(y_true, y_pred):
    """
    Calcula precisión y recall para usar en curvas.

    Returns:
    tuple: (recall, precision)
    """
    precision, recall, _ = calculate_precision_recall_f1(y_true, y_pred)
    return recall, precision

def calculate_roc_curve(y_true, y_proba, thresholds=100):
    """
    Calcula los puntos de la curva ROC.

    Returns:
    tuple: (FPRs, TPRs)
    """
    return _calculate_curve(y_true, y_proba, _tpr_fpr, thresholds)

def calculate_precision_recall_curve(y_true, y_proba, thresholds=100):
    """
    Calcula los puntos de la curva Precision-Recall.

    Returns:
    tuple: (Recalls, Precisions)
    """
    return _calculate_curve(y_true, y_proba, _precision_recall, thresholds)

def calculate_auc(x, y):
    """
    Calcula el Área Bajo la Curva (AUC) usando la regla del trapecio.

    Parameters:
    x, y (array-like): Coordenadas de la curva.

    Returns:
    float: AUC
    """
    return np.trapz(y, x)

def plot_confusion_matrix(cm):
    """
    Grafica una matriz de confusión.

    Parameters:
    cm (array-like): Matriz 2x2 con [[TN, FP], [FN, TP]].
    """
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def plot_curves(y_true, y_proba):
    """
    Grafica las curvas ROC y Precision-Recall.

    Parameters:
    y_true (array-like): Etiquetas verdaderas.
    y_proba (array-like): Probabilidades predichas para clase positiva.
    """
    fpr, tpr = calculate_roc_curve(y_true, y_proba)
    recall, precision = calculate_precision_recall_curve(y_true, y_proba)
    auc_roc = calculate_auc(fpr, tpr)
    auc_pr = calculate_auc(recall, precision)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC-ROC = {auc_roc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"AUC-PR = {auc_pr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()

def print_classification_metrics(y_true, y_pred, y_proba, label="Model"):
    """
    Muestra todas las métricas de clasificación binaria y sus gráficos.

    Parameters:
    y_true (array-like): Etiquetas verdaderas.
    y_pred (array-like): Etiquetas predichas (binarias).
    y_proba (array-like): Probabilidades predichas para clase positiva.
    label (str): Nombre del modelo o etiqueta de impresión.
    """
    cm = calculate_confusion_matrix(y_true, y_pred)
    acc = calculate_accuracy(y_true, y_pred)
    prec, rec, f1 = calculate_precision_recall_f1(y_true, y_pred)
    fpr, tpr = calculate_roc_curve(y_true, y_proba)
    recall_curve, precision_curve = calculate_precision_recall_curve(y_true, y_proba)
    auc_roc = calculate_auc(fpr, tpr)
    auc_pr = calculate_auc(recall_curve, precision_curve)

    print(f"{label} Metrics:")
    print(f"- Accuracy:   {acc:.4f}")
    print(f"- Precision:  {prec:.4f}")
    print(f"- Recall:     {rec:.4f}")
    print(f"- F1 Score:   {f1:.4f}")
    print(f"- AUC-ROC:    {auc_roc:.4f}")
    print(f"- AUC-PR:     {auc_pr:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm)
    plot_curves(y_true, y_proba)
