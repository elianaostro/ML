import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred, classes=None):
    """Calcula la matriz de confusión"""
    if classes is None:
        classes = np.unique(np.concatenate((y_true, y_pred)))
    
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i in range(len(y_true)):
        true_idx = np.where(classes == y_true[i])[0][0]
        pred_idx = np.where(classes == y_pred[i])[0][0]
        cm[true_idx, pred_idx] += 1
    
    return cm

def accuracy_score(y_true, y_pred):
    """Calcula la precisión global"""
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred, average='binary'):
    """Calcula la precisión por clase"""
    cm = confusion_matrix(y_true, y_pred)
    if average == 'binary' and cm.shape[0] == 2:
        tp = cm[1, 1]
        fp = cm[0, 1]
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    else:
        precisions = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        return np.array(precisions) if average == 'none' else np.mean(precisions)

def recall_score(y_true, y_pred, average='binary'):
    """Calcula el recall por clase"""
    cm = confusion_matrix(y_true, y_pred)
    if average == 'binary' and cm.shape[0] == 2:
        tp = cm[1, 1]
        fn = cm[1, 0]
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        recalls = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        return np.array(recalls) if average == 'none' else np.mean(recalls)

def f1_score(y_true, y_pred, average='binary'):
    """Calcula el F1-score"""
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)
    
    if average == 'binary':
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        return 2 * (precision * recall) / (precision + recall)

def roc_curve(y_true, y_proba):
    """Calcula la curva ROC"""
    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
        y_proba = y_proba[:, 1]  # Use probability of positive class (index 1)
    
    thresholds = np.sort(np.unique(y_proba))[::-1]
    tpr = []
    fpr = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape[0] == 1:  # Caso cuando solo hay una clase en las predicciones
            cm = np.vstack([cm, [0, 0]]) if y_true[0] == 1 else np.vstack([[0, 0], cm])
        
        tp = cm[1, 1] if cm.shape[0] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        fp = cm[0, 1]
        tn = cm[0, 0]
        
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    
    return np.array(fpr), np.array(tpr), thresholds

def pr_curve(y_true, y_proba):
    """Calcula la curva Precision-Recall"""
    # Extract probabilities for the positive class
    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
        y_proba = y_proba[:, 1]  # Use probability of positive class (index 1)
    
    thresholds = np.sort(np.unique(y_proba))[::-1]
    precision = []
    recall = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape[0] == 1:  # Caso cuando solo hay una clase en las predicciones
            cm = np.vstack([cm, [0, 0]]) if y_true[0] == 1 else np.vstack([[0, 0], cm])
        
        tp = cm[1, 1] if cm.shape[0] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        fp = cm[0, 1]
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision.append(prec)
        recall.append(rec)
    
    return np.array(precision), np.array(recall), thresholds

def auc(x, y):
    """
    Calcula el área bajo la curva usando la regla del trapecio
    
    Args:
        x: Valores en el eje x (array-like)
        y: Valores en el eje y (array-like)
        
    Returns:
        Área bajo la curva (float)
    """
    direction = 1 if x[-1] > x[0] else -1
    area = np.trapezoid(y, x) * direction
    return area

def plot_confusion_matrix(y_true, y_pred, classes=None, title='Confusion matrix'):
    if classes is None:
        classes = np.unique(y_true)
    """Grafica la matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred, classes)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.show()

def report_metrics(y_true, y_pred, y_proba, classes=None):
    """
    Reporta todas las métricas de performance.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        y_proba: Puntajes de predicción.
        classes: Lista de clases (opcional).
    
    Returns:
        Un diccionario con todas las métricas calculadas.
    """
    metrics = {}
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, classes)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary')
    metrics['recall'] = recall_score(y_true, y_pred, average='binary')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
    
    precision, recall, pr_thresholds = pr_curve(y_true, y_proba)
    metrics['pr_curve'] = (precision, recall, pr_thresholds)
    metrics['auc_pr'] = auc(recall, precision)
    
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    metrics['roc_curve'] = (fpr, tpr, roc_thresholds)
    metrics['auc_roc'] = auc(fpr, tpr)
    
    return metrics

def display_metrics(y_true, y_pred, y_proba, classes=None):
    """
    Calcula y muestra las métricas de performance.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        y_proba: Puntajes de predicción.
        classes: Lista de clases (opcional).
    """
    metrics = report_metrics(y_true, y_pred, y_proba, classes)
    
    print("\nAccuracy: {:.2f}".format(metrics['accuracy']))
    print("Precision: {:.2f}".format(metrics['precision']))
    print("Recall: {:.2f}".format(metrics['recall']))
    print("F1-Score: {:.2f}".format(metrics['f1_score']))
    print("\nAUC-PR: {:.2f}".format(metrics['auc_pr']))
    print("AUC-ROC: {:.2f}".format(metrics['auc_roc']))
    plot_confusion_matrix(y_true, y_pred, classes=classes)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot Precision-Recall Curve
    precision, recall, _ = metrics['pr_curve']
    axes[0].plot(recall, precision, label="PR Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].legend()
    
    # Plot ROC Curve
    fpr, tpr, _ = metrics['roc_curve']
    axes[1].plot(fpr, tpr, label="ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()