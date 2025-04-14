import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Optional, Union, Tuple, Any, Sequence 

ArrayLike = Union[List[Any], np.ndarray, pd.Series]
Numeric = Union[int, float]
Labels = Optional[Sequence[Any]]
TargetNames = Optional[List[str]]

def confusion_matrix(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    labels: Labels = None
) -> np.ndarray:
    """ 
    Computes the confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true (ArrayLike): Ground truth (correct) target values.
        y_pred (ArrayLike): Estimated targets as returned by a classifier.
        labels (Labels, optional): List of labels to index the matrix. This may be used 
            to reorder or select a subset of labels. If None is given, those that 
            appear at least once in y_true or y_pred are used in sorted order. 
            Defaults to None.

    Returns:
        np.ndarray: The confusion matrix (n_labels, n_labels) where CM[i, j] is 
                    the number of observations known to be in group i but predicted 
                    to be in group j.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if labels is None:
        present_labels = np.unique(np.concatenate((y_true_arr, y_pred_arr)))
        labels_list = sorted(list(present_labels))
    else:
        labels_list = list(labels)
        
    n_labels = len(labels_list)
    label_to_ind = {label: i for i, label in enumerate(labels_list)}
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    for true, pred in zip(y_true_arr, y_pred_arr):
        true_label_ind = label_to_ind.get(true)
        pred_label_ind = label_to_ind.get(pred)
        
        if true_label_ind is not None and pred_label_ind is not None:
            cm[true_label_ind, pred_label_ind] += 1
            
    return cm

def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculates the accuracy classification score.

    Accuracy is the proportion of correctly classified samples.

    Args:
        y_true (ArrayLike): Ground truth (correct) target values.
        y_pred (ArrayLike): Estimated targets as returned by a classifier.

    Returns:
        float: The fraction of correctly classified samples (float between 0.0 and 1.0).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("Input arrays y_true and y_pred must have the same shape.")
        
    return float(np.mean(y_true_arr == y_pred_arr))

def precision_score(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    labels: Labels = None, 
    average: Optional[str] = 'binary', 
    zero_division: Numeric = 0
) -> Union[float, np.ndarray]:
    """
    Computes the precision: tp / (tp + fp).

    Precision is the ability of the classifier not to label as positive a sample 
    that is negative.

    Args:
        y_true (ArrayLike): Ground truth target values.
        y_pred (ArrayLike): Estimated targets.
        labels (Labels, optional): The set of labels to include. Defaults to None.
        average (Optional[str], optional): Type of averaging:
            - 'binary': Only report score for the class specified by `pos_label` 
                        (assumes pos_label=1 or the second label if not specified/found).
            - 'micro': Calculate metrics globally by counting total true positives, 
                       false negatives and false positives.
            - 'macro': Calculate metrics for each label, and find their unweighted mean.
            - 'weighted': Calculate metrics for each label, find their average weighted by support.
            - None: Return the scores for each class.
            Defaults to 'binary'.
        zero_division (Numeric, optional): Value to return when precision is undefined (0/0). 
            Defaults to 0.

    Returns:
        Union[float, np.ndarray]: Precision score or array of precision scores, depending on `average`.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    
    if labels is None:
        effective_labels = sorted(list(np.unique(np.concatenate((y_true_arr, y_pred_arr)))))
    else:
        effective_labels = list(labels)

    n_labels = len(effective_labels)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    support = cm.sum(axis=1)

    if average == 'binary':
        pos_label_idx = 1
        if n_labels != 2:
             try: 
                  pos_label_idx = effective_labels.index(1) 
             except ValueError: 
                  pos_label_idx = 1 if n_labels > 1 else 0
        
        if pos_label_idx >= n_labels: return float(zero_division)
             
        tp_binary = tp[pos_label_idx]
        fp_binary = fp[pos_label_idx]
        denominator = tp_binary + fp_binary
        precision = tp_binary / denominator if denominator > 0 else float(zero_division)
        return float(precision)
        
    elif average == 'micro':
        tp_total = np.sum(tp)
        fp_total = np.sum(fp)
        denominator = tp_total + fp_total
        precision = tp_total / denominator if denominator > 0 else float(zero_division)
        return float(precision)
        
    else:
        precision_per_class = np.zeros(n_labels, dtype=float)
        denominators = tp + fp
        valid_mask = denominators > 0
        precision_per_class[valid_mask] = tp[valid_mask] / denominators[valid_mask]
        precision_per_class[~valid_mask] = float(zero_division)
            
        if average is None or average == 'none':
            return precision_per_class
        elif average == 'macro':
            return float(np.mean(precision_per_class))
        elif average == 'weighted':
            total_support = np.sum(support)
            if total_support == 0:
                 return float(zero_division)
            return float(np.average(precision_per_class, weights=support))
        else:
            raise ValueError("average parameter must be 'binary', 'micro', 'macro', 'weighted', or None")

def recall_score(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    labels: Labels = None, 
    average: Optional[str] = 'binary', 
    zero_division: Numeric = 0
) -> Union[float, np.ndarray]:
    """
    Computes the recall: tp / (tp + fn).

    Recall is the ability of the classifier to find all the positive samples.

    Args:
        y_true (ArrayLike): Ground truth target values.
        y_pred (ArrayLike): Estimated targets.
        labels (Labels, optional): The set of labels to include. Defaults to None.
        average (Optional[str], optional): Type of averaging (see precision_score). 
            Defaults to 'binary'.
        zero_division (Numeric, optional): Value to return when recall is undefined (0/0). 
            Defaults to 0.

    Returns:
        Union[float, np.ndarray]: Recall score or array of recall scores, depending on `average`.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    
    if labels is None:
        effective_labels = sorted(list(np.unique(np.concatenate((y_true_arr, y_pred_arr)))))
    else:
        effective_labels = list(labels)

    n_labels = len(effective_labels)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    if average == 'binary':
        pos_label_idx = 1
        if n_labels != 2:
             try: pos_label_idx = effective_labels.index(1)
             except ValueError: pos_label_idx = 1 if n_labels > 1 else 0
        
        if pos_label_idx >= n_labels: return float(zero_division)
             
        tp_binary = tp[pos_label_idx]
        fn_binary = fn[pos_label_idx]
        denominator = tp_binary + fn_binary
        recall = tp_binary / denominator if denominator > 0 else float(zero_division)
        return float(recall)
        
    elif average == 'micro':
        tp_total = np.sum(tp)
        fn_total = np.sum(fn) 
        denominator = tp_total + fn_total
        recall = tp_total / denominator if denominator > 0 else float(zero_division)
        return float(recall) 
        
    else:
        recall_per_class = np.zeros(n_labels, dtype=float)
        denominators = support
        valid_mask = denominators > 0
        recall_per_class[valid_mask] = tp[valid_mask] / denominators[valid_mask]
        recall_per_class[~valid_mask] = float(zero_division)
            
        if average is None or average == 'none':
            return recall_per_class
        elif average == 'macro':
            return float(np.mean(recall_per_class))
        elif average == 'weighted':
            total_support = np.sum(support)
            if total_support == 0:
                return float(zero_division)
            return float(np.average(recall_per_class, weights=support))
        else:
            raise ValueError("average parameter must be 'binary', 'micro', 'macro', 'weighted', or None")

def f1_score(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    labels: Labels = None, 
    average: Optional[str] = 'binary', 
    zero_division: Numeric = 0
) -> Union[float, np.ndarray]:
    """
    Computes the F1 score: 2 * (precision * recall) / (precision + recall).

    The F1 score can be interpreted as a harmonic mean of precision and recall.

    Args:
        y_true (ArrayLike): Ground truth target values.
        y_pred (ArrayLike): Estimated targets.
        labels (Labels, optional): The set of labels to include. Defaults to None.
        average (Optional[str], optional): Type of averaging (see precision_score). 
            Defaults to 'binary'.
        zero_division (Numeric, optional): Value to return when F1 is undefined (0/0). 
            Defaults to 0.

    Returns:
        Union[float, np.ndarray]: F1 score or array of F1 scores, depending on `average`.
    """
    # Calculate precision and recall based on the same averaging method
    precision = precision_score(y_true, y_pred, labels=labels, average=average, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, labels=labels, average=average, zero_division=zero_division)

    if isinstance(precision, np.ndarray):
        f1 = np.zeros_like(precision, dtype=float)
        denominator = precision + recall
        valid_mask = denominator > 0
        f1[valid_mask] = (2 * precision[valid_mask] * recall[valid_mask]) / denominator[valid_mask]
        f1[~valid_mask] = float(zero_division)
        return f1
    else:
        denominator = precision + recall
        f1 = (2 * precision * recall) / denominator if denominator > 0 else float(zero_division)
        return float(f1)

def roc_curve(y_true: ArrayLike, y_proba: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Receiver Operating Characteristic (ROC) curve points.

    Note: This function is primarily for binary classification or One-vs-Rest (OvR) 
          scenarios where `y_proba` represents the score for the positive class.

    Args:
        y_true (ArrayLike): True binary labels (e.g., 0 or 1).
        y_proba (ArrayLike): Target scores, can either be probability estimates of the positive 
                             class (shape [n_samples]), or confidence values (shape [n_samples]), 
                             or probability estimates for all classes (shape [n_samples, n_classes]).
                             If 2D, assumes the second column ([..., 1]) is the score for the positive class.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - fpr (np.ndarray): Increasing false positive rates.
            - tpr (np.ndarray): Increasing true positive rates.
            - thresholds (np.ndarray): Decreasing thresholds on the decision function used to compute fpr and tpr.
    """
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)

    if y_proba_arr.ndim == 2:
        if y_proba_arr.shape[1] < 2:
             raise ValueError("y_proba has 2 dimensions but fewer than 2 columns.")
        y_scores = y_proba_arr[:, 1]
    elif y_proba_arr.ndim == 1:
        y_scores = y_proba_arr
    else:
         raise ValueError("y_proba must be 1D or 2D.")
         
    pos_label_val = 1

    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores_sorted = y_scores[desc_score_indices]
    y_true_sorted = y_true_arr[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]
    thresholds = y_scores_sorted[threshold_idxs]

    tps = np.cumsum(y_true_sorted == pos_label_val)[threshold_idxs]
    fps = 1 + threshold_idxs - tps 

    n_positives = np.sum(y_true_arr == pos_label_val)
    n_negatives = y_true_arr.size - n_positives

    tpr = tps / n_positives if n_positives > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / n_negatives if n_negatives > 0 else np.zeros_like(fps, dtype=float)

    fpr = np.r_[0, fpr]
    tpr = np.r_[0, tpr]
    eps = np.finfo(thresholds.dtype).eps if len(thresholds) > 0 else 1e-6
    thresholds = np.r_[thresholds[0] + eps , thresholds]

    return fpr, tpr, thresholds

def pr_curve(y_true: ArrayLike, y_proba: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Precision-Recall curve points.

    Note: Primarily for binary classification or OvR scenarios.

    Args:
        y_true (ArrayLike): True binary labels.
        y_proba (ArrayLike): Target scores (see roc_curve args).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - precision (np.ndarray): Precision values.
            - recall (np.ndarray): Increasing recall values.
            - thresholds (np.ndarray): Decreasing thresholds used to compute precision and recall.
    """
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    
    if y_proba_arr.ndim == 2:
        if y_proba_arr.shape[1] < 2: raise ValueError("y_proba has 2 dimensions but fewer than 2 columns.")
        y_scores = y_proba_arr[:, 1]
    elif y_proba_arr.ndim == 1:
        y_scores = y_proba_arr
    else: raise ValueError("y_proba must be 1D or 2D.")

    pos_label_val = 1

    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores_sorted = y_scores[desc_score_indices]
    y_true_sorted = y_true_arr[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]
    thresholds = y_scores_sorted[threshold_idxs]
    
    tps = np.cumsum(y_true_sorted == pos_label_val)[threshold_idxs]
    pred_positives = 1 + threshold_idxs 

    n_positives = np.sum(y_true_arr == pos_label_val)

    precision = tps / pred_positives if n_positives > 0 else np.zeros_like(tps, dtype=float)
    recall = tps / n_positives if n_positives > 0 else np.zeros_like(tps, dtype=float)
    
    eps = np.finfo(thresholds.dtype).eps if len(thresholds) > 0 else 1e-6
    thresholds = np.r_[thresholds[0] + eps, thresholds] 
    
    precision = np.r_[precision[0], precision] 
    recall = np.r_[0, recall] 

    return precision, recall, thresholds

def auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Area Under the Curve (AUC) using the trapezoidal rule.

    Args:
        x (np.ndarray): X coordinates of the points defining the curve (e.g., FPR or Recall). 
                         Must be monotonic (typically increasing).
        y (np.ndarray): Y coordinates of the points defining the curve (e.g., TPR or Precision).

    Returns:
        float: The computed Area Under the Curve. Returns 0.0 if calculation fails.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return 0.0 
    
    try:
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        
        unique_x, unique_indices = np.unique(x_sorted, return_index=True)
        unique_y = y_sorted[unique_indices]

        if len(unique_x) < 2:
            return 0.0 

        area = np.trapz(unique_y, unique_x)
        return float(area)
        
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        return 0.0


def plot_confusion_matrix(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    labels: Labels = None, 
    display_labels: TargetNames = None, 
    title: str = 'Confusion Matrix', 
    cmap: Any = plt.cm.Blues, 
    ax: Optional[plt.Axes] = None 
) -> plt.Axes:
    """
    Plots the confusion matrix.

    Args:
        y_true (ArrayLike): Ground truth target values.
        y_pred (ArrayLike): Estimated targets.
        labels (Labels, optional): List of labels to index the matrix. Defaults to None.
        display_labels (TargetNames, optional): Target names used for display on axes ticks. 
            If None, `labels` are used. Defaults to None.
        title (str, optional): Title for the chart. Defaults to 'Confusion Matrix'.
        cmap (Any, optional): Colormap instance or registered colormap name. 
            Defaults to plt.cm.Blues.
        ax (Optional[plt.Axes], optional): Axes object to plot on. If None, a new figure 
            and axes are created. Defaults to None.

    Returns:
        plt.Axes: The axes object with the confusion matrix plotted.
    """

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if labels is None:
        effective_labels = sorted(list(np.unique(np.concatenate((y_true_arr, y_pred_arr)))))
    else:
        effective_labels = list(labels)
        
    if display_labels is None:
        effective_display_labels = [str(l) for l in effective_labels]
    else:
        effective_display_labels = list(display_labels)
        if len(effective_display_labels) != len(effective_labels):
             raise ValueError("Length of display_labels must match length of labels.")


    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=effective_labels)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure 

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=ax) 
    
    tick_marks = np.arange(len(effective_display_labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(effective_display_labels)
    ax.set_yticklabels(effective_display_labels)
    
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd' 
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    return ax

def calculate_metrics(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    y_proba: Optional[np.ndarray] = None,  
    labels: Labels = None, 
    target_names: TargetNames = None, 
    zero_division: Numeric = 0,
    pos_label: Any = 1
) -> Dict[str, Any]:
    """
    Calculates main classification metrics and returns them in a structured dictionary.
    For binary classification, includes additional metrics like AUC-ROC and AUC-PR when y_proba is provided.

    Args:
        y_true (ArrayLike): Ground truth target values.
        y_pred (ArrayLike): Estimated targets.
        y_proba (Optional[np.ndarray], optional): Predicted probabilities for the positive class.
            Required for AUC calculations. Defaults to None.
        labels (Labels, optional): List of labels to include. If None, uses labels present.
        target_names (TargetNames, optional): Display names for labels. If None, uses labels as strings.
        zero_division (Numeric, optional): Value for metrics when division by zero occurs. Defaults to 0.
        pos_label (Any, optional): Label to consider as positive class for binary metrics. Defaults to 1.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'classes' (Dict[str, Dict[str, float]]): Metrics (precision, recall, f1-score, support) 
                                                      for each class (using target_names as keys).
            - 'accuracy' (float): Overall accuracy.
            - 'macro avg' (Dict[str, float]): Macro averages for precision, recall, f1-score, and total support.
            - 'weighted avg' (Dict[str, float]): Weighted averages for precision, recall, f1-score, and total support.
            - Additional binary metrics if binary classification and y_proba provided:
                - 'auc_roc' (float): Area under ROC curve.
                - 'auc_pr' (float): Area under Precision-Recall curve.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    if labels is None:
        effective_labels = sorted(list(np.unique(np.concatenate((y_true_arr, y_pred_arr)))))
    else:
        effective_labels = list(labels)

    if target_names is None:
        effective_target_names = [str(l) for l in effective_labels]
    else:
        effective_target_names = list(target_names)
        if len(effective_labels) != len(effective_target_names):
             raise ValueError("Length of labels and target_names must match.")
             
    is_binary = len(effective_labels) == 2

    p = precision_score(y_true_arr, y_pred_arr, labels=effective_labels, average=None, zero_division=zero_division)
    r = recall_score(y_true_arr, y_pred_arr, labels=effective_labels, average=None, zero_division=zero_division)
    f1 = f1_score(y_true_arr, y_pred_arr, labels=effective_labels, average=None, zero_division=zero_division)
    
    if is_binary:
        precision = precision_score(y_true_arr, y_pred_arr, average='binary', zero_division=zero_division)
        recall = recall_score(y_true_arr, y_pred_arr, average='binary', zero_division=zero_division)
        f1_value = f1_score(y_true_arr, y_pred_arr, average='binary', zero_division=zero_division)
    else:
        precision = precision_score(y_true_arr, y_pred_arr, average='weighted', zero_division=zero_division)
        recall = recall_score(y_true_arr, y_pred_arr, average='weighted', zero_division=zero_division)
        f1_value = f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=zero_division)
    
    true_counts = Counter(y_true_arr)
    support = np.array([true_counts.get(l, 0) for l in effective_labels])

    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    total_support = np.sum(support)
    
    macro_p = np.average(p)
    macro_r = np.average(r)
    macro_f1 = np.average(f1)
    
    weighted_p = np.average(p, weights=support) if total_support > 0 else float(zero_division)
    weighted_r = np.average(r, weights=support) if total_support > 0 else float(zero_division)
    weighted_f1 = np.average(f1, weights=support) if total_support > 0 else float(zero_division)

    metrics_dict: Dict[str, Any] = {
        "classes": {}, 
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_value,
        "macro avg": {
            "precision": macro_p, "recall": macro_r, "f1-score": macro_f1, "support": total_support
        },
        "weighted avg": {
            "precision": weighted_p, "recall": weighted_r, "f1-score": weighted_f1, "support": total_support
        }
    }
    
    for i, name in enumerate(effective_target_names):
        metrics_dict["classes"][name] = {
            "precision": p[i], "recall": r[i], "f1-score": f1[i], "support": support[i]
        }
    if is_binary and y_proba is not None:
        y_proba_arr = np.asarray(y_proba)
        
        
        if y_proba_arr.ndim == 2 and y_proba_arr.shape[1] >= 2:
            try:
                pos_idx = effective_labels.index(pos_label)
                y_scores = y_proba_arr[:, pos_idx]
            except ValueError:
                y_scores = y_proba_arr[:, 1]
        else:
            y_scores = y_proba_arr
            
        y_true_binary = (y_true_arr == pos_label).astype(int)
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
        metrics_dict["auc_roc"] = auc(fpr, tpr)
        
        precision_curve, recall_curve, _ = pr_curve(y_true_binary, y_scores)
        metrics_dict["auc_pr"] = auc(recall_curve, precision_curve)
        
    return metrics_dict


def print_classification_report(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    labels: Labels = None, 
    target_names: TargetNames = None, 
    zero_division: Numeric = 0
) -> None:
    """
    Calculates and prints a text summary of the main classification metrics.

    Args:
        y_true (ArrayLike): Ground truth target values.
        y_pred (ArrayLike): Estimated targets.
        labels (Labels, optional): List of labels to include. If None, uses labels present.
        target_names (TargetNames, optional): Display names for labels. If None, uses labels as strings.
        zero_division (Numeric, optional): Value for metrics when division by zero occurs. Defaults to 0.
    """
    metrics = calculate_metrics(y_true, y_pred, labels=labels, target_names=target_names, zero_division=zero_division)

    class_names_in_dict = list(metrics["classes"].keys())
    if not class_names_in_dict:
        print("Warning: No classes to report.")
        return

    headers = ["precision", "recall", "f1-score", "support"]
    name_width = max(len(name) for name in class_names_in_dict)
    avg_width = len("weighted avg")
    acc_width = len("accuracy")
    width = max(name_width, avg_width, acc_width) 
    
    title = "Classification Report"
    report_str = f"\n{title}\n{'=' * len(title)}\n"
    
    header_fmt = '{:>{width}s} ' + ' {:>9s}' * len(headers)
    report_str += header_fmt.format('', *headers, width=width) + '\n\n'

    row_fmt = '{:>{width}s} ' + ' {:>9.2f}' * 3 + ' {:>9d}\n'
    for name in class_names_in_dict:
        class_metrics = metrics["classes"][name]
        report_str += row_fmt.format(name,
                                     class_metrics["precision"],
                                     class_metrics["recall"],
                                     class_metrics["f1-score"],
                                     int(class_metrics["support"]), 
                                     width=width)

    report_str += '\n' 
    
    acc_fmt = '{:>{width}s} ' + ' {:>9s}' * 2 + ' {:>9.2f} {:>9d}\n'
    total_support_int = int(metrics["macro avg"]["support"]) # Get total support once
    report_str += acc_fmt.format('accuracy', '', '', metrics["accuracy"], total_support_int, width=width)
    
    avg_fmt = '{:>{width}s} ' + ' {:>9.2f}' * 3 + ' {:>9d}\n'
    macro_metrics = metrics["macro avg"]
    report_str += avg_fmt.format('macro avg', 
                                 macro_metrics["precision"], macro_metrics["recall"], 
                                 macro_metrics["f1-score"], total_support_int, width=width)
    
    weighted_metrics = metrics["weighted avg"]
    report_str += avg_fmt.format('weighted avg', 
                                 weighted_metrics["precision"], weighted_metrics["recall"], 
                                 weighted_metrics["f1-score"], total_support_int, width=width)
    
    report_str += "=" * (width + 1 + 9 * len(headers) + len(headers)) + "\n" # Footer line

    print(report_str)


def display_full_metrics(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    y_proba: Optional[np.ndarray] = None, 
    labels: Labels = None, 
    target_names: TargetNames = None, 
    pos_label: Any = 1,
    title_suffix: str = ""
) -> None:
    """
    Displays a comprehensive summary of classification performance metrics.

    Includes the text classification report, the confusion matrix plot, and 
    (if `y_proba` is provided) ROC and Precision-Recall curve plots.
    Automatically handles binary vs. multi-class scenarios for plotting curves.

    Args:
        y_true (ArrayLike): Ground truth target values.
        y_pred (ArrayLike): Estimated targets.
        y_proba (Optional[np.ndarray], optional): Predicted probabilities 
            (shape [n_samples, n_classes] or [n_samples] for binary positive class). 
            Required for ROC/PR curves. Defaults to None.
        labels (Labels, optional): List of labels to include and order. If None, uses labels present.
        target_names (TargetNames, optional): Display names for labels. If None, uses labels as strings.
        pos_label (Any, optional): The label considered as the positive class in binary scenarios 
            for ROC/PR curves. Defaults to 1.
        title_suffix (str, optional): Suffix to append to plot titles. Defaults to "".
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if labels is None:
        effective_labels = sorted(list(np.unique(np.concatenate((y_true_arr, y_pred_arr)))))
    else:
        effective_labels = list(labels)
    n_classes = len(effective_labels)
    
    if target_names is None:
        effective_target_names = [str(l) for l in effective_labels]
    else:
        effective_target_names = list(target_names)
        if len(effective_labels) != len(effective_target_names):
            raise ValueError("Length of labels and target_names must match.")

    print_classification_report(y_true_arr, y_pred_arr, labels=effective_labels, target_names=effective_target_names)

    cm_title = f"Confusion Matrix {title_suffix}".strip()
    print(f"\n--- {cm_title} ---")

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5)) 
    plot_confusion_matrix(y_true_arr, y_pred_arr, labels=effective_labels, 
                            display_labels=effective_target_names, title=cm_title, ax=ax_cm)
    fig_cm.tight_layout() 
    plt.show()

    if y_proba is not None:
        y_proba_arr = np.asarray(y_proba)
        print(f"\n--- Performance Curves {title_suffix} ---".strip())
        
        fig_curves, axes_curves = plt.subplots(1, 2, figsize=(14, 6))
        cmap_curves = plt.cm.get_cmap('tab10', max(10, n_classes)) 

        is_binary = n_classes == 2

        if is_binary:
            print(f"Binary case detected (Classes: {effective_target_names}). Positive label assumed: '{pos_label}'")
            y_scores: Optional[np.ndarray] = None
            if y_proba_arr.ndim == 2:
                if y_proba_arr.shape[1] == n_classes:
                     try:
                         pos_label_idx = effective_labels.index(pos_label)
                         y_scores = y_proba_arr[:, pos_label_idx]
                     except ValueError:
                         print(f"Warning: pos_label '{pos_label}' not found in labels '{effective_labels}'. Using scores from column 1.")
                         pos_label_idx = 1 
                         y_scores = y_proba_arr[:, pos_label_idx] 
                else:
                     print(f"Warning: y_proba is 2D but shape {y_proba_arr.shape} doesn't match n_classes={n_classes}. Cannot plot curves.")
            elif y_proba_arr.ndim == 1:
                 y_scores = y_proba_arr 
            else:
                 print(f"Warning: y_proba has unexpected shape {y_proba_arr.shape}. Cannot plot curves.")

            if y_scores is not None:
                y_true_binary = (y_true_arr == pos_label).astype(int)
                
                fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
                roc_auc = auc(fpr, tpr)
                axes_curves[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
                axes_curves[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
                axes_curves[0].set_title(f'ROC Curve {title_suffix}'.strip())
                axes_curves[0].set_xlabel('False Positive Rate (FPR)')
                axes_curves[0].set_ylabel('True Positive Rate (TPR)')
                axes_curves[0].legend(loc="lower right")

                precision, recall, _ = pr_curve(y_true_binary, y_scores)
                pr_auc = auc(recall, precision) 
                baseline = np.sum(y_true_binary) / len(y_true_binary) if len(y_true_binary) > 0 else 0
                axes_curves[1].plot(recall, precision, color='blue', lw=2, label=f'PR (AUC = {pr_auc:.2f})')
                axes_curves[1].axhline(baseline, color='grey', lw=1, linestyle='--', label=f'Baseline ({baseline:.2f})')
                axes_curves[1].set_title(f'Precision-Recall Curve {title_suffix}'.strip())
                axes_curves[1].set_xlabel('Recall')
                axes_curves[1].set_ylabel('Precision')
                axes_curves[1].legend(loc="lower left")

        else:
            print("Multi-class case detected. Plotting One-vs-Rest (OvR) curves.")
            if y_proba_arr.ndim != 2 or y_proba_arr.shape[1] != n_classes:
                 print(f"Error: Expected y_proba shape (n_samples, {n_classes}), got {y_proba_arr.shape}. Cannot plot OvR curves.")
                 plt.close(fig_curves) 
                 return 

            axes_curves[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
            print("AUC ROC (OvR):")
            for i, label in enumerate(effective_labels):
                y_true_binary = (y_true_arr == label)
                if np.sum(y_true_binary) == 0 or np.sum(y_true_binary) == len(y_true_arr): 
                    print(f"  - Skipping Class '{effective_target_names[i]}' (only one class present in y_true for OvR).")
                    continue 
                
                y_scores = y_proba_arr[:, i]
                fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
                roc_auc = auc(fpr, tpr)
                print(f"  - Class '{effective_target_names[i]}': {roc_auc:.3f}")
                axes_curves[0].plot(fpr, tpr, color=cmap_curves(i % cmap_curves.N), lw=2,
                                     label=f'{effective_target_names[i]} (AUC = {roc_auc:.2f})')
            axes_curves[0].set_title(f'ROC Curves (OvR) {title_suffix}'.strip())
            axes_curves[0].set_xlabel('False Positive Rate (FPR)')
            axes_curves[0].set_ylabel('True Positive Rate (TPR)')
            axes_curves[0].legend(loc="lower right", fontsize='small')

            print("\nAUC PR (OvR):")
            for i, label in enumerate(effective_labels):
                y_true_binary = (y_true_arr == label)
                if np.sum(y_true_binary) == 0 or np.sum(y_true_binary) == len(y_true_arr): continue

                y_scores = y_proba_arr[:, i]
                precision, recall, _ = pr_curve(y_true_binary, y_scores)
                pr_auc = auc(recall, precision) 
                print(f"  - Class '{effective_target_names[i]}': {pr_auc:.3f}")
                baseline = np.sum(y_true_binary) / len(y_true_binary) if len(y_true_binary) > 0 else 0
                axes_curves[1].plot(recall, precision, color=cmap_curves(i % cmap_curves.N), lw=2, 
                                     label=f'{effective_target_names[i]} (AUC = {pr_auc:.2f})')
                axes_curves[1].plot([0, 1], [baseline, baseline], color=cmap_curves(i % cmap_curves.N), lw=1, linestyle=':') 

            axes_curves[1].set_title(f'Precision-Recall Curves (OvR) {title_suffix}'.strip())
            axes_curves[1].set_xlabel('Recall')
            axes_curves[1].set_ylabel('Precision')
            axes_curves[1].legend(loc="lower left", fontsize='small')

        for ax in axes_curves:
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.grid(True)

        fig_curves.tight_layout()
        plt.show()