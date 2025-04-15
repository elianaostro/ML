import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred, classes=None):
    """Calcula la matriz de confusión"""
    if classes is None:
        classes = np.unique(np.concatenate((y_true, y_pred)))
    
    n_classes = len(classes)
    # Vectorized implementation using numpy's bincount
    true_indices = np.searchsorted(classes, y_true)
    pred_indices = np.searchsorted(classes, y_pred)
    
    cm = np.bincount(
        true_indices * n_classes + pred_indices,
        minlength=n_classes**2
    ).reshape(n_classes, n_classes)
    
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
    
    # Vectorized calculation for all classes
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    precisions = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    
    return precisions if average == 'none' else np.mean(precisions)

def recall_score(y_true, y_pred, average='binary'):
    """Calcula el recall por clase"""
    cm = confusion_matrix(y_true, y_pred)
    
    if average == 'binary' and cm.shape[0] == 2:
        tp = cm[1, 1]
        fn = cm[1, 0]
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Vectorized calculation for all classes
    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp
    recalls = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    
    return recalls if average == 'none' else np.mean(recalls)

def f1_score(y_true, y_pred, average='binary'):
    """Calcula el F1-score"""
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)
    
    if average == 'binary':
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Vectorized calculation for multiclass
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 * (precision * recall) / (precision + recall)
        f1[np.isnan(f1)] = 0
    return f1 if average == 'none' else np.mean(f1)

def roc_curve(y_true, y_scores):
    """Calcula la curva ROC"""
    thresholds = np.unique(y_scores)[::-1]  # np.unique already sorts
    tpr = np.zeros_like(thresholds, dtype=float)
    fpr = np.zeros_like(thresholds, dtype=float)
    
    # Vectorized calculation for each threshold
    for i, thresh in enumerate(thresholds):
        y_pred = (y_scores >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape[0] == 1:
            cm = np.vstack([cm, [0, 0]]) if y_true[0] == 1 else np.vstack([[0, 0], cm])
        
        tp = cm[1, 1] if cm.shape[0] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        fp = cm[0, 1]
        tn = cm[0, 0]
        
        tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return fpr, tpr, thresholds

def pr_curve(y_true, y_scores):
    """Calcula la curva Precision-Recall"""
    thresholds = np.unique(y_scores)[::-1]
    precision = np.zeros_like(thresholds, dtype=float)
    recall = np.zeros_like(thresholds, dtype=float)
    
    # Vectorized calculation for each threshold
    for i, thresh in enumerate(thresholds):
        y_pred = (y_scores >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape[0] == 1:
            cm = np.vstack([cm, [0, 0]]) if y_true[0] == 1 else np.vstack([[0, 0], cm])
        
        tp = cm[1, 1] if cm.shape[0] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        fp = cm[0, 1]
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall, thresholds

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
    area = np.trapz(y, x) * direction  # Changed to trapz (more standard)
    return area

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix'):
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
    # Vectorized text placement
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.show()