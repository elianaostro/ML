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

def roc_curve(y_true, y_scores):
    """Calcula la curva ROC"""
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tpr = []
    fpr = []
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
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

def pr_curve(y_true, y_scores):
    """Calcula la curva Precision-Recall"""
    thresholds = np.sort(np.unique(y_scores))[::-1]
    precision = []
    recall = []
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
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
    """Calcula el área bajo la curva usando la regla del trapecio"""
    direction = 1 if x[-1] > x[0] else -1
    area = np.trapz(y, x) * direction
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
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.show()