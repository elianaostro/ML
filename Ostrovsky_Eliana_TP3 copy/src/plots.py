import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple

def plot_class_distribution(y):
    """
    Plots the distribution of classes in the dataset.
    
    Args:
        y (np.ndarray): Array with class labels
    """
    plt.figure(figsize=(12, 6))
    unique, counts = np.unique(y, return_counts=True)
    
    sorted_indices = np.argsort(counts)[::-1]
    unique = unique[sorted_indices]
    counts = counts[sorted_indices]
    
    plt.bar(unique, counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print(f"Most common class: {unique[0]} with {counts[0]} samples")
    print(f"Least common class: {unique[-1]} with {counts[-1]} samples")
    print(f"Class imbalance ratio (most common / least common): {counts[0]/counts[-1]:.2f}")

def plot_class_averages(X, y, num_classes=None):
    """
    Plots the average image for each class.
    
    Args:
        X (np.ndarray): Array of images
        y (np.ndarray): Array of labels
        num_classes (int, optional): Number of classes to show. If None, shows all classes.
    """
    classes = np.unique(y)
    if num_classes is not None:
        classes = classes[:num_classes]
    
    n_cols = int(np.sqrt(num_classes))  
    n_rows = (len(classes) + n_cols - 1) // n_cols 
    
    plt.figure(figsize=(15, 2 * n_rows))
    
    for i, cls in enumerate(classes):
        class_images = X[y == cls]
        avg_image = np.mean(class_images, axis=0).reshape(28, 28)
        
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(avg_image, cmap='gray')
        plt.title(f'Class {cls}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_samples(X: np.ndarray, y: np.ndarray, num_samples: int = 5, 
                     num_classes: Optional[int] = None, figsize: Tuple[int, int] = (15, 15)) -> None:
    """
    Visualize random samples from each class.
    
    Args:
        X: Array of images.
        y: Array of labels.
        num_samples: Number of samples to display per class.
        num_classes: Number of classes to display. If None, display all.
        figsize: Figure size (width, height) in inches.
    """
    classes = np.unique(y)
    if num_classes is not None:
        classes = classes[:num_classes]
    
    plt.figure(figsize=figsize)
    for i, cls in enumerate(classes):
        indices = np.where(y == cls)[0]
        
        if len(indices) >= num_samples:
            selected_indices = np.random.choice(indices, num_samples, replace=False)
        else:
            selected_indices = indices
        
        for j, idx in enumerate(selected_indices):
            plt.subplot(len(classes), num_samples, i * num_samples + j + 1)
            plt.imshow(X[idx].reshape(28, 28), cmap='gray')
            plt.axis('off')
            if j == 0: 
                plt.title(f'Class {cls}')
    
    plt.tight_layout()
    plt.show()

def plot_learning_curves(history: Dict[str, List[float]], figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Plot learning curves for loss and accuracy.
    
    Args:
        history: Dictionary containing training and validation metrics.
        figsize: Figure size (width, height) in inches.
    """
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Cross-Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history and history['val_accuracy']:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(confusion_matrix: np.ndarray, figsize: Tuple[int, int] = (10, 8), 
                         class_subset: Optional[List[int]] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix to plot.
        figsize: Figure size (width, height) in inches.
        class_subset: Subset of classes to plot. If None, plot all classes.
    """
    if class_subset is not None:
        cm = confusion_matrix[class_subset, :][:, class_subset]
    else:
        cm = confusion_matrix
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    num_classes = cm.shape[0]
    if num_classes <= 20:  
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


def plot_experiment_results(results: List[Dict[str, Any]], metric: str = 'val_accuracy', x_lim: Optional[Tuple[int, int]] = None) -> None:
    """
    Plot experiment results for comparison.
    
    Args:
        results: List of experiment results.
        metric: Metric to compare ('val_accuracy', 'val_loss', 'train_accuracy', 'train_loss', 'accuracy_val_vs_train').
        x_lim: Tuple specifying the x-axis limits (start, end). If None, no limits are applied.
    """
    plt.figure(figsize=(12, 8))
    
    for res in results:
        if metric in res['history']:
            plt.plot(res['history'][metric], label=res['name'])
        elif metric == 'accuracy_val_vs_train':
            plt.plot(res['history']['val_accuracy'], label=f"{res['name']} - Val Accuracy")
            plt.plot(res['history']['train_accuracy'], label=f"{res['name']} - Train Accuracy")
    
    if x_lim is not None:
        plt.xlim(x_lim)
    
    plt.title(f"Comparison of {metric.replace('_', ' ').title()}")
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title() if metric != 'accuracy_val_vs_train' else 'Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def compare_training_times(results: List[Dict[str, Any]]) -> None:
    """
    Plot training times for different experiments.
    
    Args:
        results: List of experiment results.
    """
    names = [res['name'] for res in results]
    times = [res['training_time'] for res in results]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, times)
    
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{time_val:.2f}s", ha='center')
    
    plt.title('Training Time Comparison')
    plt.xlabel('Experiment')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def compare_final_metrics(results: List[Dict[str, Any]]) -> None:
    """
    Plot final metrics for different experiments.
    
    Args:
        results: List of experiment results.
    """
    names = [res['name'] for res in results]
    train_acc = [res['final_train_accuracy'] for res in results]
    val_acc = [res['final_val_accuracy'] for res in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, train_acc, width, label='Train Accuracy')
    bars2 = plt.bar(x + width/2, val_acc, width, label='Validation Accuracy')
    
    for bar, val in zip(bars1, train_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', fontsize=8)
    
    for bar, val in zip(bars2, val_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', fontsize=8)
    
    plt.title('Final Accuracy Comparison')
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()