
# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

def plot_learning_curves(history: Dict[str, List[float]], figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Plot learning curves for loss and accuracy.
    
    Args:
        history: Dictionary containing training and validation metrics.
        figsize: Figure size (width, height) in inches.
    """
    plt.figure(figsize=figsize)
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Cross-Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot accuracy
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
    if num_classes <= 20:  # Only show tick marks if not too many classes
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

def plot_class_examples(X_images, y_images, figsize=(13, 15), suptitle="Examples of each character class"):
    """
    Plot one example of each class in the dataset.
    
    Parameters:
    -----------
    X_images : numpy.ndarray
        The image data, shape (n_samples, n_features)
    y_images : numpy.ndarray
        The labels, shape (n_samples,)
    figsize : tuple, optional
        Figure size, default (13, 15)
    suptitle : str, optional
        Title for the entire figure
    """
    unique_labels = np.unique(y_images)
    num_classes = len(unique_labels)
    selected_images = []
    selected_labels = []

    for label in unique_labels:
        idx = np.where(y_images == label)[0][0]
        selected_images.append(X_images[idx].reshape(28, 28))
        selected_labels.append(label)

    n_cols = int(np.sqrt(num_classes))  
    n_rows = (num_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  

    for i, (img, label) in enumerate(zip(selected_images, selected_labels)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Class: {label}")
        axes[i].axis('off')

    for j in range(num_classes, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.suptitle(suptitle, y=0.95, fontsize=18)
    plt.subplots_adjust(top=0.92)
    plt.show()
    
