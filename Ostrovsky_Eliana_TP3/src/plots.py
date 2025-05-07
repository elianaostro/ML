import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 1. Visualizar la distribución de clases mediante un histograma
def plot_class_distribution(y):
    """
    Plots the distribution of classes in the dataset.
    
    Args:
        y (np.ndarray): Array with class labels
    """
    plt.figure(figsize=(12, 6))
    unique, counts = np.unique(y, return_counts=True)
    
    # Sort by frequency
    sorted_indices = np.argsort(counts)[::-1]
    unique = unique[sorted_indices]
    counts = counts[sorted_indices]
    
    # Plot horizontal bar chart
    plt.barh(unique, counts)
    plt.xlabel('Count')
    plt.ylabel('Class')
    plt.title('Class Distribution')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Print the most and least common classes
    print(f"Most common class: {unique[0]} with {counts[0]} samples")
    print(f"Least common class: {unique[-1]} with {counts[-1]} samples")
    print(f"Class imbalance ratio (most common / least common): {counts[0]/counts[-1]:.2f}")

# 2. Analizar estadísticas básicas de las imágenes
def analyze_image_statistics(X):
    """
    Analyzes basic image statistics across the dataset.
    
    Args:
        X (np.ndarray): Array of images
    """
    # Reshape to (n_samples, pixel_count) if needed
    if len(X.shape) > 2:
        X_flat = X.reshape(X.shape[0], -1)
    else:
        X_flat = X
    
    # Calculate statistics per image
    pixel_means = np.mean(X_flat, axis=1)
    pixel_stds = np.std(X_flat, axis=1)
    pixel_mins = np.min(X_flat, axis=1)
    pixel_maxs = np.max(X_flat, axis=1)
    
    # Plot histograms of these statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].hist(pixel_means, bins=50)
    axes[0, 0].set_title('Distribution of Mean Pixel Values')
    axes[0, 0].set_xlabel('Mean Pixel Value')
    axes[0, 0].set_ylabel('Count')
    
    axes[0, 1].hist(pixel_stds, bins=50)
    axes[0, 1].set_title('Distribution of Pixel Standard Deviations')
    axes[0, 1].set_xlabel('Pixel Standard Deviation')
    axes[0, 1].set_ylabel('Count')
    
    axes[1, 0].hist(pixel_mins, bins=50)
    axes[1, 0].set_title('Distribution of Minimum Pixel Values')
    axes[1, 0].set_xlabel('Min Pixel Value')
    axes[1, 0].set_ylabel('Count')
    
    axes[1, 1].hist(pixel_maxs, bins=50)
    axes[1, 1].set_title('Distribution of Maximum Pixel Values')
    axes[1, 1].set_xlabel('Max Pixel Value')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Average mean pixel value: {np.mean(pixel_means):.4f}")
    print(f"Average pixel standard deviation: {np.mean(pixel_stds):.4f}")
    print(f"Proportion of images with completely black pixels: {np.mean(pixel_mins == 0):.4f}")
    print(f"Proportion of images with completely white pixels: {np.mean(pixel_maxs == 255):.4f}")

# 3. Visualizar el "promedio" de cada clase
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
    
    n_cols = 5  # Number of columns in the grid
    n_rows = (len(classes) + n_cols - 1) // n_cols  # Ceiling division
    
    plt.figure(figsize=(15, 3 * n_rows))
    
    for i, cls in enumerate(classes):
        # Get all images of this class
        class_images = X[y == cls]
        
        # Calculate average image
        avg_image = np.mean(class_images, axis=0).reshape(28, 28)
        
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(avg_image, cmap='gray')
        plt.title(f'Class {cls}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 4. Analizar la complejidad de las imágenes (usando varianza o conteo de bordes)
def analyze_image_complexity(X, y):
    """
    Analyzes image complexity by class using pixel variance.
    
    Args:
        X (np.ndarray): Array of images
        y (np.ndarray): Array of labels
    """
    classes = np.unique(y)
    complexity_by_class = []
    
    for cls in classes:
        # Get all images of this class
        class_images = X[y == cls]
        
        # Calculate variance for each image (higher variance = more complex)
        variances = np.var(class_images.reshape(class_images.shape[0], -1), axis=1)
        
        # Store average complexity for this class
        complexity_by_class.append((cls, np.mean(variances)))
    
    # Sort by complexity
    complexity_by_class.sort(key=lambda x: x[1], reverse=True)
    
    # Plot
    classes_sorted = [x[0] for x in complexity_by_class]
    complexity_sorted = [x[1] for x in complexity_by_class]
    
    plt.figure(figsize=(12, 6))
    plt.bar(classes_sorted, complexity_sorted)
    plt.xlabel('Class')
    plt.ylabel('Average Pixel Variance (Complexity)')
    plt.title('Image Complexity by Class')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Print the most and least complex classes
    print(f"Most complex class: {complexity_by_class[0][0]} with variance {complexity_by_class[0][1]:.4f}")
    print(f"Least complex class: {complexity_by_class[-1][0]} with variance {complexity_by_class[-1][1]:.4f}")

# 5. Visualizar ejemplos de cada clase en una cuadrícula
def plot_examples_by_class(X, y, examples_per_class=5):
    """
    Plots multiple examples for each class in a grid.
    
    Args:
        X (np.ndarray): Array of images
        y (np.ndarray): Array of labels
        examples_per_class (int): Number of examples to show per class
    """
    classes = np.unique(y)
    num_classes_to_show = min(10, len(classes))  # Limit to 10 classes for readability
    
    plt.figure(figsize=(15, 2 * num_classes_to_show))
    
    for i, cls in enumerate(classes[:num_classes_to_show]):
        # Get images for this class
        indices = np.where(y == cls)[0]
        
        # Select random examples
        if len(indices) >= examples_per_class:
            selected_indices = np.random.choice(indices, examples_per_class, replace=False)
        else:
            selected_indices = indices
        
        # Plot examples
        for j, idx in enumerate(selected_indices):
            plt.subplot(num_classes_to_show, examples_per_class, i * examples_per_class + j + 1)
            plt.imshow(X[idx].reshape(28, 28), cmap='gray')
            plt.axis('off')
            
            if j == 0:  # Add class label to the first example
                plt.title(f'Class {cls}')
    
    plt.tight_layout()
    plt.show()






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

def visualize_samples(X: np.ndarray, y: np.ndarray, num_samples: int = 5, 
                     num_classes: Optional[int] = None, figsize: Tuple[int, int] = (15, 10)) -> None:
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
        # Get indices for this class
        indices = np.where(y == cls)[0]
        
        # Select random samples
        if len(indices) >= num_samples:
            selected_indices = np.random.choice(indices, num_samples, replace=False)
        else:
            selected_indices = indices
        
        # Plot samples
        for j, idx in enumerate(selected_indices):
            plt.subplot(len(classes), num_samples, i * num_samples + j + 1)
            plt.imshow(X[idx].reshape(28, 28), cmap='gray')
            plt.axis('off')
            if j == 0:  # Add class label to the first image
                plt.title(f'Class {cls}')
    
    plt.tight_layout()
    plt.show()