# src/preprocessing.py
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union, Dict, Any


# Note: knn_value is kept for reference but NOT used by the vectorized handle_missing_values
def knn_value(
    base_row: pd.Series, 
    df: pd.DataFrame, 
    target_col: str, 
    feature_cols: List[str], 
    k: int
) -> Any:
    """
    Calculates the imputed value for a single row using k-nearest neighbors (KNN).
    Finds the k closest rows in `df` based on `feature_cols` and returns the 
    most common value of `target_col` among those neighbors.

    Note: This function processes one row at a time and is less efficient than 
          vectorized approaches for imputing many values. It's superseded by
          the vectorized KNN logic in `handle_missing_values`.

    Args:
        base_row (pd.Series): The row containing the missing value and features 
                              to compare distances with.
        df (pd.DataFrame): The DataFrame containing potential neighbors (rows with 
                           known values in `target_col`).
        target_col (str): The name of the column to impute.
        feature_cols (List[str]): List of column names to use for calculating 
                                  Euclidean distance. Assumes these columns are 
                                  present and numeric in both `base_row` and `df`.
        k (int): The number of nearest neighbors to consider.

    Returns:
        Any: The most frequent value (mode) of `target_col` among the k nearest 
             neighbors. Returns np.nan if no neighbors are found or if the mode 
             cannot be determined (e.g., all neighbors have NaN in `target_col`).
    """
    if df.empty or not feature_cols:
        return np.nan # Cannot compute if no neighbors or no features

    df_copy = df.copy()
    
    # Calculate Euclidean distance (vectorized)
    # Ensure base_row features are numeric and aligned
    base_features = base_row[feature_cols].astype(float).values
    neighbor_features = df_copy[feature_cols].astype(float).values
    
    # Handle potential NaNs in features used for distance calculation
    # Option 1: Drop rows with NaNs in features (simplest)
    # valid_neighbor_mask = ~np.isnan(neighbor_features).any(axis=1)
    # neighbor_features = neighbor_features[valid_neighbor_mask]
    # df_copy = df_copy[valid_neighbor_mask]
    # if neighbor_features.shape[0] < k: return np.nan # Not enough valid neighbors
    # Option 2: Impute NaNs in features before distance (e.g., with mean) - more complex
    
    # Assuming no NaNs in features for simplicity here, matching original code's implicit assumption
    distances = np.linalg.norm(neighbor_features - base_features, axis=1)
    df_copy['distance'] = distances
    
    # Find k smallest distances
    # Ensure k is not larger than the number of available neighbors
    actual_k = min(k, len(df_copy))
    if actual_k == 0:
        return np.nan
        
    nearest_neighbors = df_copy.nsmallest(actual_k, 'distance')
    
    # Calculate the mode of the target column among neighbors
    # Drop NaNs in target column before calculating mode
    target_values = nearest_neighbors[target_col].dropna()
    if target_values.empty:
        return np.nan # All neighbors had NaN in target column
        
    mode_result = target_values.mode()
    
    # Return the first mode if multiple exist, or NaN if empty
    return mode_result[0] if not mode_result.empty else np.nan


def normalize(
    X: pd.DataFrame, 
    means: Optional[pd.Series] = None, 
    stds: Optional[pd.Series] = None, 
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Normalizes (standardizes) numerical columns in a DataFrame (Z-score scaling).

    Subtracts the mean and divides by the standard deviation for each selected column.
    Can use pre-computed means and stds (e.g., from a training set) for consistent 
    scaling on a validation/test set.

    Args:
        X (pd.DataFrame): The input DataFrame to normalize.
        means (Optional[pd.Series], optional): Pre-computed means (indexed by column name). 
            If None, means are calculated from X. Defaults to None.
        stds (Optional[pd.Series], optional): Pre-computed standard deviations (indexed by 
            column name). If None, stds are calculated from X. Defaults to None.
        exclude_cols (Optional[List[str]], optional): List of column names to exclude 
            from normalization. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]: A tuple containing:
            - X_norm (pd.DataFrame): The DataFrame with specified columns normalized.
            - calculated_means (pd.Series): The means used for normalization (either 
              provided or calculated).
            - calculated_stds (pd.Series): The standard deviations used for normalization 
              (either provided or calculated).
    """
    X_norm = X.copy()
    
    if exclude_cols is None:
        exclude_cols = []
        
    # Identify numerical columns to normalize (excluding specified ones)
    numeric_cols = X.select_dtypes(include=np.number).columns
    cols_to_normalize = numeric_cols.difference(exclude_cols)
    
    # Calculate means and stds if not provided
    calculated_means = means if means is not None else X[cols_to_normalize].mean()
    calculated_stds = stds if stds is not None else X[cols_to_normalize].std()
    
    # Add small epsilon to std deviation to prevent division by zero
    epsilon = 1e-8
    stds_safe = calculated_stds + epsilon
    
    # Apply normalization (vectorized)
    X_norm[cols_to_normalize] = (X[cols_to_normalize] - calculated_means) / stds_safe
    
    # Fill NaNs that might result from 0 std dev columns (if epsilon wasn't enough or mean was NaN)
    # X_norm[cols_to_normalize] = X_norm[cols_to_normalize].fillna(0) # Optional: fill NaNs post-normalization
    
    return X_norm, calculated_means, calculated_stds


def handle_missing_values(
    df: pd.DataFrame, 
    strategy: str = 'mean', 
    numeric_strategy: Optional[str] = None,
    categorical_strategy: Optional[str] = None,
    fill_value_numeric: float = 0.0,
    fill_value_categorical: str = 'Missing',
    knn_k: int = 5,
    knn_cols_numeric: Optional[List[str]] = None,
    train_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Handles missing values (NaN) in a DataFrame using various strategies, 
    including vectorized KNN imputation for numerical columns.

    Strategies can be specified globally or separately for numeric/categorical columns.

    Args:
        df (pd.DataFrame): DataFrame with potential missing values.
        strategy (str, optional): Default strategy if specific ones aren't set. 
            Options: 'mean', 'median', 'mode', 'zero', 'constant', 'knn', 'remove_row'. 
            Defaults to 'mean'.
        numeric_strategy (Optional[str], optional): Strategy specifically for numeric columns. 
            Overrides `strategy`. Options: 'mean', 'median', 'zero', 'constant', 'knn', 'remove_row'. 
            Defaults to None (uses `strategy`).
        categorical_strategy (Optional[str], optional): Strategy specifically for non-numeric 
            (categorical/object) columns. Overrides `strategy`. Options: 'mode', 'constant', 
            'remove_row'. Defaults to None (uses `strategy`).
        fill_value_numeric (float, optional): Constant value used if numeric_strategy is 'constant'. 
            Defaults to 0.0.
        fill_value_categorical (str, optional): Constant value used if categorical_strategy is 'constant'. 
            Defaults to 'Missing'.
        knn_k (int, optional): Number of neighbors for KNN imputation (if used). Defaults to 5.
        knn_cols_numeric (Optional[List[str]], optional): Specific numeric columns to apply KNN imputation to. 
            If None, applies to all numeric columns with NaNs when strategy is 'knn'. Defaults to None.
        train_df (Optional[pd.DataFrame], optional): Reference DataFrame (e.g., training set) 
            to calculate statistics (mean, median, mode) or find neighbors from, ensuring consistency. 
            If None, statistics/neighbors are based on the input `df` itself. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with missing values handled according to the specified strategies.

    Raises:
        ValueError: If an invalid strategy is provided or KNN requirements are not met.
    """
    df_imputed = df.copy()
    
    # Determine effective strategies
    num_strat = numeric_strategy if numeric_strategy is not None else strategy
    cat_strat = categorical_strategy if categorical_strategy is not None else strategy
    
    # Use train_df for fitting statistics if provided, otherwise use df itself
    reference_df = train_df if train_df is not None else df_imputed

    # --- Handle Numeric Columns ---
    numeric_cols = df_imputed.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df_imputed[col].isnull().any(): # Process only columns with NaNs
            if num_strat == 'mean':
                fill_val = reference_df[col].mean()
                df_imputed[col].fillna(fill_val, inplace=True)
            elif num_strat == 'median':
                fill_val = reference_df[col].median()
                df_imputed[col].fillna(fill_val, inplace=True)
            elif num_strat == 'zero':
                 df_imputed[col].fillna(0.0, inplace=True)
            elif num_strat == 'constant':
                 df_imputed[col].fillna(fill_value_numeric, inplace=True)
            elif num_strat == 'knn':
                # KNN imputation is handled collectively below
                pass 
            elif num_strat == 'remove_row':
                 # Handled globally at the end
                 pass
            else:
                 raise ValueError(f"Invalid numeric strategy: {num_strat}")

    # --- Vectorized KNN Imputation (if selected) ---
    if num_strat == 'knn':
        print(f"Applying vectorized KNN imputation (k={knn_k}) using mean of neighbors...")
        
        # Columns to apply KNN to
        target_knn_cols = knn_cols_numeric if knn_cols_numeric else numeric_cols
        target_knn_cols = [col for col in target_knn_cols if col in df_imputed.columns and df_imputed[col].isnull().any()]

        if not target_knn_cols:
             print("No numeric columns with NaNs found for KNN imputation.")
        else:
            # Feature columns to use for distance (all numeric cols except the target)
            # Consider if normalization should happen *before* KNN for better distance metrics
            all_numeric_features = df_imputed.select_dtypes(include=np.number).columns
            
            # Prepare reference data (potential neighbors) - rows without NaNs in feature columns
            # For simplicity, let's use rows from reference_df that are complete *in all numeric features*
            # A more robust approach might use only features relevant to the current target column.
            complete_rows_mask_ref = reference_df[all_numeric_features].notna().all(axis=1)
            reference_features_np = reference_df.loc[complete_rows_mask_ref, all_numeric_features].values
            
            if reference_features_np.shape[0] < knn_k:
                 print(f"Warning: Fewer than k={knn_k} complete rows available in reference data ({reference_features_np.shape[0]}). Cannot perform KNN imputation. Falling back to mean.")
                 # Fallback to mean imputation for remaining NaNs
                 for col in target_knn_cols:
                     if df_imputed[col].isnull().any():
                          fill_val = reference_df[col].mean()
                          df_imputed[col].fillna(fill_val, inplace=True)
                 target_knn_cols = [] # Mark as done

            for col in target_knn_cols:
                # Indices of rows in the *current* df with NaN in this column
                nan_in_col_mask = df_imputed[col].isnull()
                nan_rows_indices = df_imputed.index[nan_in_col_mask]

                if not nan_rows_indices.any(): continue # Skip if no NaNs left in this column

                # Features to use for distance (excluding the target column itself)
                dist_feature_cols = [f for f in all_numeric_features if f != col]
                
                if not dist_feature_cols: 
                    print(f"Warning: No features available for KNN distance calculation for column '{col}'. Falling back to mean.")
                    fill_val = reference_df[col].mean()
                    df_imputed.loc[nan_rows_indices, col] = fill_val
                    continue

                # Get features of rows needing imputation (handling potential NaNs in *their* features)
                # Option: Impute features first, or use nan-robust distance, or use only complete feature rows.
                # Let's use mean imputation for features of rows needing imputation *before* distance calculation.
                target_features_df = df_imputed.loc[nan_rows_indices, dist_feature_cols]
                feature_means = reference_df[dist_feature_cols].mean() # Use means from reference
                target_features_df = target_features_df.fillna(feature_means)
                target_features_np = target_features_df.values

                # Get corresponding features and target values from reference data
                reference_dist_features_np = reference_df.loc[complete_rows_mask_ref, dist_feature_cols].values
                reference_target_values_np = reference_df.loc[complete_rows_mask_ref, col].values

                # Calculate pairwise distances (vectorized)
                # Using broadcasting: (n_target, 1, d) - (1, n_ref, d) -> (n_target, n_ref, d) -> sum -> sqrt
                # Ensure no NaNs remain in features before distance calculation
                if np.isnan(target_features_np).any() or np.isnan(reference_dist_features_np).any():
                     print(f"Warning: NaNs found in feature matrices for KNN distance (col '{col}'). Check feature imputation. Falling back to mean.")
                     fill_val = reference_df[col].mean()
                     df_imputed.loc[nan_rows_indices, col] = fill_val
                     continue
                     
                distances = np.sqrt(np.sum((target_features_np[:, np.newaxis, :] - reference_dist_features_np[np.newaxis, :, :])**2, axis=2))

                # Find indices of the k nearest neighbors for each target row
                # Adjust k if fewer reference points than k
                actual_k = min(knn_k, reference_dist_features_np.shape[0])
                # Use argpartition for efficiency if only k smallest needed, then sort the k
                # Or use argsort and take the first k
                knn_indices = np.argsort(distances, axis=1)[:, :actual_k] 

                # Get the target values of the neighbors
                neighbor_values = reference_target_values_np[knn_indices] # Shape (n_target, k)

                # Calculate the mean of neighbor values (ignoring NaNs among neighbors)
                # Note: If using mode, replace np.nanmean with apply_along_axis(_mode_1d, ...)
                with np.errstate(invalid='ignore'): # Suppress mean of empty slice warning
                    imputed_values = np.nanmean(neighbor_values, axis=1)

                # Handle cases where all neighbors had NaN (nanmean returns NaN)
                # Fallback to global mean for these cases
                fallback_mask = np.isnan(imputed_values)
                if np.any(fallback_mask):
                     global_mean = reference_df[col].mean()
                     imputed_values[fallback_mask] = global_mean
                     print(f"Warning: Some KNN imputations for '{col}' failed (all neighbors NaN?). Used global mean as fallback.")

                # Fill NaNs in the original DataFrame
                df_imputed.loc[nan_rows_indices, col] = imputed_values


    # --- Handle Categorical Columns ---
    categorical_cols = df_imputed.select_dtypes(exclude=np.number).columns
    for col in categorical_cols:
        if df_imputed[col].isnull().any():
             if cat_strat == 'mode':
                 fill_val = reference_df[col].mode()[0] if not reference_df[col].mode().empty else fill_value_categorical # Handle empty mode
                 df_imputed[col].fillna(fill_val, inplace=True)
             elif cat_strat == 'constant':
                 df_imputed[col].fillna(fill_value_categorical, inplace=True)
             elif cat_strat == 'remove_row':
                 # Handled globally at the end
                 pass
             # 'mean', 'median', 'knn' are not applicable to categorical
             elif cat_strat in ['mean', 'median', 'knn', 'zero']:
                  print(f"Warning: Strategy '{cat_strat}' not suitable for categorical column '{col}'. Using 'constant' ('{fill_value_categorical}').")
                  df_imputed[col].fillna(fill_value_categorical, inplace=True)
             else:
                  raise ValueError(f"Invalid categorical strategy: {cat_strat}")

    # --- Global Row Removal (if selected for either type) ---
    if num_strat == 'remove_row' or cat_strat == 'remove_row':
        df_imputed.dropna(inplace=True)

    return df_imputed


def split_data(
    df: pd.DataFrame, 
    target_column: str, 
    train_ratio: float = 0.8, 
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Splits a DataFrame into random train and validation sets.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the column containing the target variable.
        train_ratio (float, optional): The proportion of the dataset to include 
            in the training split. Defaults to 0.8.
        random_state (Optional[int], optional): Seed for the random number 
            generator for reproducible shuffling. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: A tuple containing:
            - X_train (pd.DataFrame): Training features.
            - X_val (pd.DataFrame): Validation features.
            - y_train (np.ndarray): Training target values.
            - y_val (np.ndarray): Validation target values.
    """
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.")
        
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Determine split index
    train_size = int(train_ratio * len(df_shuffled))
    
    # Split data
    df_train = df_shuffled.iloc[:train_size]
    df_val = df_shuffled.iloc[train_size:]
    
    # Separate features and target
    X_train = df_train.drop(columns=[target_column])
    X_val = df_val.drop(columns=[target_column])
    y_train = df_train[target_column].values
    y_val = df_val[target_column].values
    
    return X_train, X_val, y_train, y_val


def stratified_split(
    X: Union[pd.DataFrame, np.ndarray], 
    y: np.ndarray, 
    test_ratio: float = 0.2, 
    random_state: Optional[int] = None
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], np.ndarray, np.ndarray]:
    """
    Performs a stratified split of data into training and test sets.

    Maintains the same proportion of classes in both the train and test sets
    as in the original dataset.

    Args:
        X (Union[pd.DataFrame, np.ndarray]): Features data. Can be Pandas DataFrame 
                                             or NumPy array.
        y (np.ndarray): Target labels array corresponding to X.
        test_ratio (float, optional): Proportion of the dataset to include in the 
                                      test split. Defaults to 0.2.
        random_state (Optional[int], optional): Seed for reproducibility. Defaults to None.

    Returns:
        Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], np.ndarray, np.ndarray]: 
        A tuple containing:
            - X_train: Training features.
            - X_test: Test features.
            - y_train: Training target labels.
            - y_test: Test target labels.
            (Type of X_train/X_test matches input type of X).
    """
    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio must be between 0 and 1.")
        
    if random_state is not None:
        np.random.seed(random_state)
    
    classes, y_indices = np.unique(y, return_inverse=True)
    n_samples = len(y)
    n_classes = len(classes)
    
    # Get indices for each class
    class_indices: Dict[int, np.ndarray] = {
        i: np.where(y_indices == i)[0] for i in range(n_classes)
    }
    
    train_indices: List[int] = []
    test_indices: List[int] = []
    
    # Split indices within each class
    for i in range(n_classes):
        indices_for_class = class_indices[i]
        n_class_samples = len(indices_for_class)
        n_test_class = max(1, int(np.round(n_class_samples * test_ratio))) # Ensure at least 1 test sample per class if possible
        n_train_class = n_class_samples - n_test_class

        if n_train_class < 0: # Should not happen with max(1, ...) but as safety
             n_train_class = 0
             n_test_class = n_class_samples
        
        # Shuffle indices within the class
        np.random.shuffle(indices_for_class)
        
        # Assign indices to train and test sets
        train_indices.extend(indices_for_class[:n_train_class])
        test_indices.extend(indices_for_class[n_train_class:])
    
    # Shuffle the final train and test indices to mix classes
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Return splits based on input type
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else: # Assuming NumPy array
        X_train = X[train_indices]
        X_test = X[test_indices]
        
    y_train = y[train_indices]
    y_test = y[test_indices]
           
    return X_train, X_test, y_train, y_test


def split_and_normalize(
    df: pd.DataFrame, 
    target_column: str, 
    exclude_cols: Optional[List[str]] = None, 
    train_ratio: float = 0.8, 
    imputation_strategy: str = 'mean', # Added imputation control
    knn_k: int = 5,                   # Added imputation control
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Splits data into train/validation, handles missing values (post-split), 
    and normalizes features based on the training set.

    Args:
        df (pd.DataFrame): DataFrame to split, impute, and normalize.
        target_column (str): Name of the target column.
        exclude_cols (Optional[List[str]], optional): Columns to exclude from normalization. 
            Defaults to None. Target column is always excluded implicitly.
        train_ratio (float, optional): Proportion of data for training. Defaults to 0.8.
        imputation_strategy (str, optional): Strategy for handle_missing_values 
            (applied after split). Defaults to 'mean'.
        knn_k (int, optional): k value if imputation_strategy is 'knn'. Defaults to 5.
        random_state (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]: 
        A tuple containing:
            - X_train_norm (pd.DataFrame): Imputed and normalized training features.
            - X_val_norm (pd.DataFrame): Imputed and normalized validation features.
            - y_train (np.ndarray): Training target values.
            - y_val (np.ndarray): Validation target values.
            - means (pd.Series): Means used for normalization (from training set).
            - stds (pd.Series): Standard deviations used for normalization (from training set).
    """
    # 1. Split the data (randomly)
    X_train, X_val, y_train, y_val = split_data(df, target_column, train_ratio, random_state)
    
    # 2. Handle missing values - fit on train, transform both train and val
    #    Use the selected strategy. For KNN, use train_df=X_train in the call for X_val.
    X_train = handle_missing_values(X_train, strategy=imputation_strategy, knn_k=knn_k, train_df=None) # Impute train based on itself
    X_val = handle_missing_values(X_val, strategy=imputation_strategy, knn_k=knn_k, train_df=X_train) # Impute val based on train stats/neighbors

    # 3. Normalize - fit on train, transform both train and val
    # Ensure target column is not in exclude_cols implicitly
    if exclude_cols is None:
        exclude_cols_norm = []
    else:
        exclude_cols_norm = exclude_cols.copy()
    # exclude_cols_norm.append(target_column) # Not needed as target is already separated

    X_train_norm, means, stds = normalize(X_train, exclude_cols=exclude_cols_norm)
    # Apply train means/stds to validation set
    X_val_norm, _, _ = normalize(X_val, means=means, stds=stds, exclude_cols=exclude_cols_norm)
    
    return X_train_norm, X_val_norm, y_train, y_val, means, stds


def stratified_split_and_normalize(
    df: pd.DataFrame, 
    target_column: str, 
    exclude_cols: Optional[List[str]] = None, 
    test_ratio: float = 0.2, 
    imputation_strategy: str = 'mean', # Added imputation control
    knn_k: int = 5,                   # Added imputation control
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Performs stratified split, handles missing values (post-split), and normalizes features.

    Splits data into train/test sets maintaining class proportions, imputes missing 
    values using statistics/neighbors from the training set, and normalizes features 
    based on the training set.

    Args:
        df (pd.DataFrame): DataFrame to split, impute, and normalize.
        target_column (str): Name of the target column.
        exclude_cols (Optional[List[str]], optional): Columns to exclude from normalization. 
            Defaults to None.
        test_ratio (float, optional): Proportion of data for the test set. Defaults to 0.2.
        imputation_strategy (str, optional): Strategy for handle_missing_values 
            (applied after split). Defaults to 'mean'.
        knn_k (int, optional): k value if imputation_strategy is 'knn'. Defaults to 5.
        random_state (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]: 
        A tuple containing:
            - X_train_norm (pd.DataFrame): Imputed and normalized training features.
            - X_test_norm (pd.DataFrame): Imputed and normalized test features.
            - y_train (np.ndarray): Training target values.
            - y_test (np.ndarray): Test target values.
            - means (pd.Series): Means used for normalization (from training set).
            - stds (pd.Series): Standard deviations used for normalization (from training set).
    """
    # 1. Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column].values
    
    # 2. Split the data (stratified)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_ratio, random_state)
    
    # 3. Handle missing values - fit on train, transform both train and test
    X_train = handle_missing_values(X_train, strategy=imputation_strategy, knn_k=knn_k, train_df=None) # Impute train based on itself
    X_test = handle_missing_values(X_test, strategy=imputation_strategy, knn_k=knn_k, train_df=X_train) # Impute test based on train stats/neighbors

    # 4. Normalize - fit on train, transform both train and test
    if exclude_cols is None:
        exclude_cols_norm = []
    else:
        exclude_cols_norm = exclude_cols.copy()
        
    X_train_norm, means, stds = normalize(X_train, exclude_cols=exclude_cols_norm)
    X_test_norm, _, _ = normalize(X_test, means=means, stds=stds, exclude_cols=exclude_cols_norm)
    
    return X_train_norm, X_test_norm, y_train, y_test, means, stds