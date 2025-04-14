import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Any

def plot_decision_boundary(
    model: Any, 
    X: np.ndarray, 
    y: np.ndarray, 
    title: str = "Decision Boundary", 
    step: float = 0.02, 
    figsize: Tuple[float, float] = (10, 6),
    cmap_background = plt.cm.RdYlBu, 
    cmap_points = plt.cm.viridis  
) -> None:
    """
    Plots the decision boundary of a trained 2D classifier model.

    Creates a mesh grid, predicts the class for each point on the grid, 
    and plots the resulting regions along with the original data points.

    Args:
        model (Model): A trained classifier model object that has a `predict` method.
        X (np.ndarray): Feature data (n_samples, 2). Must have exactly 2 features.
        y (np.ndarray): True labels corresponding to X (n_samples,).
        title (str, optional): Title for the plot. Defaults to "Decision Boundary".
        step (float, optional): Step size for the mesh grid resolution. Defaults to 0.02.
        figsize (Tuple[float, float], optional): Figure size (width, height). Defaults to (10, 6).
        cmap_background: Colormap for the decision regions. Defaults to RdYlBu.
        cmap_points: Colormap for the data points. Defaults to Viridis.

    Raises:
        ValueError: If X does not have exactly 2 features.
    """
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Input feature data X must have exactly 2 columns (shape n_samples, 2). Got shape {X.shape}")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    
    try:
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid_points)
        try:
            Z = Z.astype(float) 
        except ValueError:
             # Try mapping if predict returns non-numeric labels? Requires classes_
             # Or raise error if predict output isn't suitable for plotting
             print("Warning: Model predictions could not be cast to float for plotting.")
             # For now, proceed assuming numeric or handle error
             pass 
        Z = Z.reshape(xx.shape)
    except Exception as e:
         print(f"Error during model prediction for decision boundary: {e}")
         return # Abort plotting if prediction fails

    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot decision regions using contourf
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.6)
    
    # Plot data points, colored by true label
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolor='k', s=30, alpha=0.8)
    
    # Add labels and title
    plt.title(title)
    # Attempt to get feature names if X is a DataFrame? For now, generic names.
    plt.xlabel("Feature 1") 
    plt.ylabel("Feature 2")
    
    # Set plot limits
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    plt.show()


def plot_feature_importance(
    model: Any, 
    feature_names: List[str], 
    title: str = "Feature Importance",
    figsize: Tuple[float, float] = (10, 6)
) -> None:
    """
    Plots feature importances for models that expose a `feature_importances_` attribute 
    (e.g., RandomForest, DecisionTree from this project, or scikit-learn models).

    Args:
        model (Model): A trained model instance having a `feature_importances_` attribute 
                       (typically a NumPy array).
        feature_names (List[str]): List of names corresponding to the features, in the 
                                   same order as used by the model.
        title (str, optional): Title for the plot. Defaults to "Feature Importance".
        figsize (Tuple[float, float], optional): Figure size. Defaults to (10, 6).

    Raises:
        AttributeError: If the model does not have a `feature_importances_` attribute.
        ValueError: If the number of feature names does not match the number of importances.
    """
    # Check if the model has the required attribute
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError("The provided model does not have a 'feature_importances_' attribute.")
    
    importances = model.feature_importances_
    
    # Ensure importances is a 1D array
    if importances.ndim != 1:
         raise ValueError(f"Expected model.feature_importances_ to be a 1D array, but got shape {importances.shape}")
         
    n_features = len(importances)
    
    # Check if number of feature names matches
    if len(feature_names) != n_features:
        raise ValueError(f"Number of feature names ({len(feature_names)}) does not match "
                         f"number of feature importances ({n_features}).")

    # Sort features by importance (descending)
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_feature_names = [feature_names[i] for i in indices]

    # Create the plot
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.bar(range(n_features), sorted_importances, align="center")
    
    # Add feature names as x-axis ticks
    plt.xticks(range(n_features), sorted_feature_names, rotation=90)
    
    # Adjust plot appearance
    plt.xlim([-1, n_features]) # Add padding to x-axis limits
    plt.ylabel("Importance Score")
    plt.xlabel("Feature")
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()