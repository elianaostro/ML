
def plot_decision_boundary(
    model: Model, 
    X: np.ndarray, 
    y: np.ndarray, 
    title: str = "Decision Boundary", 
    step: float = 0.02, 
    figsize: Tuple[float, float] = (10, 6),
    cmap_background = plt.cm.RdYlBu, # Colormap for background
    cmap_points = plt.cm.viridis    # Colormap for points
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

    # Define plot boundaries based on data range, adding a margin
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    
    # Predict class for each point on the grid
    # Flatten mesh grid for prediction, then reshape result
    try:
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid_points)
        # Ensure Z contains numerical labels suitable for contourf
        # Attempt conversion if needed, assuming model.predict returns labels
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

def plot_learning_curve(
    model: Model, 
    X: np.ndarray, 
    y: np.ndarray, 
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5), 
    cv: int = 5, # Number of cross-validation folds
    scoring: Callable[[np.ndarray, np.ndarray], float] = accuracy_score, # Scoring function
    scoring_name: str = "Accuracy", # Name for the y-axis label
    random_state: Optional[int] = None,
    figsize: Tuple[float, float] = (8, 6)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates and plots a learning curve for a given model.

    Evaluates model performance (using the specified scoring function) on training 
    and validation sets for different training set sizes, using cross-validation.

    Args:
        model (Model): The model instance to evaluate (must have fit and predict methods).
        X (np.ndarray): Feature data (n_samples, n_features).
        y (np.ndarray): Target labels (n_samples,).
        train_sizes (np.ndarray, optional): Relative or absolute numbers of training examples 
            to use. Defaults to 5 steps between 10% and 100%.
        cv (int, optional): Number of cross-validation folds to use for evaluating 
            each training size. Defaults to 5.
        scoring (Callable[[np.ndarray, np.ndarray], float], optional): Function to evaluate 
            performance (e.g., accuracy_score, f1_score). Defaults to accuracy_score.
        scoring_name (str, optional): Display name for the score used on the y-axis. 
            Defaults to "Accuracy".
        random_state (Optional[int], optional): Seed for shuffling data in cross-validation. 
            Defaults to None.
        figsize (Tuple[float, float], optional): Figure size. Defaults to (8, 6).


    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - train_sizes_abs (np.ndarray): Absolute number of training samples used for each point.
            - mean_train_scores (np.ndarray): Average training score across CV folds for each size.
            - mean_val_scores (np.ndarray): Average validation score across CV folds for each size.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    
    # Convert relative train_sizes to absolute numbers
    if train_sizes.dtype == float and np.all(train_sizes <= 1.0):
        train_sizes_abs = (train_sizes * n_samples).astype(int)
    else:
        train_sizes_abs = train_sizes.astype(int)
        
    # Ensure sizes are unique and within valid range [1, n_samples]
    train_sizes_abs = np.unique(train_sizes_abs)
    train_sizes_abs = train_sizes_abs[(train_sizes_abs > 0) & (train_sizes_abs <= n_samples)]
    
    if len(train_sizes_abs) == 0:
        raise ValueError("No valid training sizes found after processing `train_sizes`.")

    # Store scores for each size
    all_train_scores: List[List[float]] = []
    all_val_scores: List[List[float]] = []

    # Iterate through each training size
    for size in train_sizes_abs:
        fold_train_scores: List[float] = []
        fold_val_scores: List[float] = []
        
        # Perform cross-validation for the current size
        # Using simple random permutation split here instead of KFold for illustration
        # A KFold or StratifiedKFold object could be used for more robust CV
        for i in range(cv):
            # Create a random split for this fold and size
            indices = np.random.permutation(n_samples)
            # Use first 'size' indices for training, rest for validation *within this fold*
            # Note: A more standard CV approach splits all data into K folds first.
            # This approach samples 'size' train and n-size val repeatedly.
            train_idx, val_idx = indices[:size], indices[size:] 

            # Check if validation set is empty (can happen if size=n_samples)
            if len(val_idx) == 0:
                # Handle case: e.g., skip fold, assign NaN, or use full train set as val
                # Assigning NaN is often reasonable.
                fold_train_scores.append(np.nan)
                fold_val_scores.append(np.nan)
                # print(f"Warning: Validation set empty for train_size={size}, fold={i}. Assigning NaN.")
                continue

            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            # Train the model on the training subset for this fold
            # Clone the model if it modifies state? Assume fit resets internal state.
            try:
                 model.fit(X_train_fold, y_train_fold)
            except Exception as e:
                 print(f"Error fitting model for train_size={size}, fold={i}: {e}")
                 # Append NaN if fit fails
                 fold_train_scores.append(np.nan)
                 fold_val_scores.append(np.nan)
                 continue


            # Evaluate on training data for this fold
            train_pred = model.predict(X_train_fold)
            fold_train_scores.append(scoring(y_train_fold, train_pred))

            # Evaluate on validation data for this fold
            val_pred = model.predict(X_val_fold)
            fold_val_scores.append(scoring(y_val_fold, val_pred))
            
        # Store scores for this training size
        all_train_scores.append(fold_train_scores)
        all_val_scores.append(fold_val_scores)

    # Calculate mean and standard deviation across folds for each size
    # Use nanmean/nanstd to handle potential NaNs from skipped folds
    mean_train_scores = np.nanmean(all_train_scores, axis=1)
    std_train_scores = np.nanstd(all_train_scores, axis=1)
    mean_val_scores = np.nanmean(all_val_scores, axis=1)
    std_val_scores = np.nanstd(all_val_scores, axis=1)

    # --- Plot the Learning Curve ---
    plt.figure(figsize=figsize)
    plt.title("Learning Curve")
    plt.xlabel("Number of Training Samples")
    plt.ylabel(scoring_name)
    plt.grid(True)

    # Plot mean scores
    plt.plot(train_sizes_abs, mean_train_scores, 'o-', color="r", label="Training score")
    plt.plot(train_sizes_abs, mean_val_scores, 'o-', color="g", label="Cross-validation score")

    # Plot score variance bands (optional)
    plt.fill_between(train_sizes_abs, mean_train_scores - std_train_scores,
                     mean_train_scores + std_train_scores, alpha=0.1, color="r")
    plt.fill_between(train_sizes_abs, mean_val_scores - std_val_scores,
                     mean_val_scores + std_val_scores, alpha=0.1, color="g")

    plt.legend(loc="best")
    plt.show()

    return train_sizes_abs, mean_train_scores, mean_val_scores


def plot_feature_importance(
    model: Model, 
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