# src/preprocessing.py
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union, Dict, Any

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cleaning rules specific to the dataset structure.

    This includes:
    - Clipping or setting specific columns to NaN based on valid ranges.
    - Setting negative numeric values to NaN.
    - Applying IQR-based outlier detection (marking as NaN using 5th/95th percentiles).
    - Removing rows with too many NaN values (>= 7).

    Args:
        df (pd.DataFrame): The input DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.copy()

    # Apply valid range constraints using np.where
    df_cleaned['CellAdhesion'] = np.where(
        df_cleaned['CellAdhesion'].between(0, 1, inclusive='both') | df_cleaned['CellAdhesion'].isna(),
        df_cleaned['CellAdhesion'], np.nan
    )
    df_cleaned['NuclearMembrane'] = np.where(
        df_cleaned['NuclearMembrane'].between(1, 5, inclusive='both') | df_cleaned['NuclearMembrane'].isna(),
        df_cleaned['NuclearMembrane'], np.nan
    )
    df_cleaned['OxygenSaturation'] = np.where(
        df_cleaned['OxygenSaturation'].between(0, 100, inclusive='both') | df_cleaned['OxygenSaturation'].isna(),
        df_cleaned['OxygenSaturation'], np.nan
    )
    df_cleaned['Vascularization'] = np.where(
        df_cleaned['Vascularization'].between(0, 10, inclusive='both') | df_cleaned['Vascularization'].isna(),
        df_cleaned['Vascularization'], np.nan
    )
    df_cleaned['InflammationMarkers'] = np.where(
        df_cleaned['InflammationMarkers'].between(0, 100, inclusive='both') | df_cleaned['InflammationMarkers'].isna(),
        df_cleaned['InflammationMarkers'], np.nan
    )

    # Vectorized check for negative numbers in all numeric columns
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
    df_numeric = df_cleaned[numeric_cols]
    df_cleaned[numeric_cols] = df_numeric.where(df_numeric >= 0, np.nan)

    # IQR Outlier Detection (setting outliers to NaN using 5th/95th percentiles)
    q05 = df_cleaned[numeric_cols].quantile(0.05)
    q95 = df_cleaned[numeric_cols].quantile(0.95)
    iqr_pseudo = q95 - q05 

    lower_bound = q05 - 1.5 * iqr_pseudo
    upper_bound = q95 + 1.5 * iqr_pseudo

    for column in numeric_cols:
        col_data = df_cleaned[column]
        is_outlier = (col_data < lower_bound[column]) | (col_data > upper_bound[column])
        df_cleaned[column] = col_data.where(~is_outlier, np.nan)

    # Remove rows with too many NaNs
    nan_limit = 7
    df_cleaned['NaN_Count'] = df_cleaned.isna().sum(axis=1)
    df_cleaned = df_cleaned[df_cleaned['NaN_Count'] < nan_limit]
    df_cleaned = df_cleaned.drop(columns=['NaN_Count']) 

    return df_cleaned

def handle_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles specific categorical features in the DataFrame.

    - Converts 'CellType' into two binary columns ('Epthlial', 'Mesnchymal') 
      via one-hot encoding (handling potential NaNs).
    - Converts 'GeneticMutation' ('Presnt'/'Absnt') into a single binary column.
    - Drops the original 'CellType' column.

    Args:
        df (pd.DataFrame): The input DataFrame with categorical features.

    Returns:
        pd.DataFrame: The DataFrame with categorical features processed.
    """
    df_processed = df.copy()
    
    df_processed['Epthlial'] = np.where(df_processed['CellType'].isna(), np.nan,
                                       (df_processed['CellType'] == 'Epthlial').astype(float))
    df_processed['Mesnchymal'] = np.where(df_processed['CellType'].isna(), np.nan,
                                         (df_processed['CellType'] == 'Mesnchymal').astype(float))
                                         
    df_processed['GeneticMutation'] = (df_processed['GeneticMutation'] == 'Presnt').astype(int)
    
    df_processed = df_processed.drop(columns=["CellType"])
    
    return df_processed

def detect_outliers_iqr(X: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Detects outliers in a numerical NumPy array using the Interquartile Range (IQR) method.

    Args:
        X (np.ndarray): Input data array (n_samples, n_features). Assumed numerical.
        factor (float, optional): The multiplier for the IQR to determine outlier bounds. 
                                  Typically 1.5. Defaults to 1.5.

    Returns:
        np.ndarray: A boolean array of the same shape as X, where True indicates 
                    an outlier.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1) 
        
    q1 = np.nanpercentile(X, 25, axis=0) 
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return (X < lower_bound) | (X > upper_bound)

# Removed the knn_value function as requested.

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
            - calculated_means (pd.Series): The means used for normalization.
            - calculated_stds (pd.Series): The standard deviations used for normalization.
    """
    X_norm = X.copy()
    
    if exclude_cols is None:
        exclude_cols = []
        
    numeric_cols = X.select_dtypes(include=np.number).columns
    cols_to_normalize = numeric_cols.difference(exclude_cols)
    
    calculated_means = means if means is not None else X[cols_to_normalize].mean()
    calculated_stds = stds if stds is not None else X[cols_to_normalize].std()
    
    epsilon = 1e-8
    stds_safe = calculated_stds + epsilon
    
    X_norm[cols_to_normalize] = (X[cols_to_normalize] - calculated_means) / stds_safe
    
    return X_norm, calculated_means, calculated_stds

def handle_missing_values(
    df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Handles missing values (NaN) in numerical columns of a DataFrame using mean imputation.

    Assumes the input DataFrame contains only columns to be treated numerically 
    for the purpose of imputation.

    Args:
        df (pd.DataFrame): DataFrame with potential missing values in numerical columns.
        train_df (Optional[pd.DataFrame], optional): Reference DataFrame (e.g., training set) 
            to calculate means from, ensuring consistency. If None, means are 
            calculated from the input `df` itself. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with missing values in numerical columns imputed using the mean.
    """
    df_imputed = df.copy()
    
    # Use train_df for calculating means if provided, otherwise use df itself
    reference_df = train_df if train_df is not None else df_imputed

    # --- Handle Numeric Columns (ALWAYS USE MEAN) ---
    # Select only numeric columns for imputation
    numeric_cols = df_imputed.select_dtypes(include=np.number).columns
    
    if not numeric_cols.empty: # Proceed only if there are numeric columns
        for col in numeric_cols:
            if df_imputed[col].isnull().any(): 
                # Calculate mean from reference_df for the current column
                fill_val = reference_df[col].mean()
                # Check if mean calculation resulted in NaN (e.g., all NaNs in reference column)
                if pd.isna(fill_val):
                     # Fallback: Impute with 0 or raise error/warning
                     fill_val = 0.0 
                     print(f"Warning: Mean for column '{col}' could not be calculated (all NaNs?). Imputing with 0.0.")
                
                # --- FIX APPLIED HERE ---
                # Fill NaNs by assigning the result back, avoiding inplace=True on a potential copy
                df_imputed[col] = df_imputed[col].fillna(fill_val)
                # --- END FIX ---

    # Note: Categorical handling and row removal were removed in the previous step
    # based on the assumption of only numeric columns for imputation.

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
        
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    train_size = int(train_ratio * len(df_shuffled))
    
    df_train = df_shuffled.iloc[:train_size]
    df_val = df_shuffled.iloc[train_size:]
    
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
        A tuple containing: X_train, X_test, y_train, y_test.
    """
    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio must be between 0 and 1.")
        
    if random_state is not None:
        np.random.seed(random_state)
    
    classes, y_indices = np.unique(y, return_inverse=True)
    n_samples = len(y)
    n_classes = len(classes)
    
    class_indices: Dict[int, np.ndarray] = {
        i: np.where(y_indices == i)[0] for i in range(n_classes)
    }
    
    train_indices: List[int] = []
    test_indices: List[int] = []
    
    for i in range(n_classes):
        indices_for_class = class_indices[i]
        n_class_samples = len(indices_for_class)
        n_test_class = max(1, int(np.round(n_class_samples * test_ratio))) 
        n_train_class = n_class_samples - n_test_class

        if n_train_class < 0: 
             n_train_class = 0
             n_test_class = n_class_samples
        
        np.random.shuffle(indices_for_class)
        
        train_indices.extend(indices_for_class[:n_train_class])
        test_indices.extend(indices_for_class[n_train_class:])
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else: 
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
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Splits data into train/validation, handles missing values in numeric columns 
    using mean imputation, and normalizes features based on the training set.

    Assumes input DataFrame primarily contains numerical features after target separation.

    Args:
        df (pd.DataFrame): DataFrame to split, impute, and normalize.
        target_column (str): Name of the target column.
        exclude_cols (Optional[List[str]], optional): Columns to exclude from normalization. 
            Defaults to None. 
        train_ratio (float, optional): Proportion of data for training. Defaults to 0.8.
        random_state (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]: 
        A tuple containing: X_train_norm, X_val_norm, y_train, y_val, means, stds.
    """
    # 1. Split the data (randomly)
    X_train, X_val, y_train, y_val = split_data(df, target_column, train_ratio, random_state)
    
    # 2. Handle missing values (mean imputation for numeric columns)
    X_train = handle_missing_values(X_train, train_df=None) # Impute train based on itself
    X_val = handle_missing_values(X_val, train_df=X_train)  # Impute val based on train mean

    # 3. Normalize - fit on train, transform both
    if exclude_cols is None:
        exclude_cols_norm = []
    else:
        exclude_cols_norm = exclude_cols.copy()

    X_train_norm, means, stds = normalize(X_train, exclude_cols=exclude_cols_norm)
    X_val_norm, _, _ = normalize(X_val, means=means, stds=stds, exclude_cols=exclude_cols_norm)
    
    return X_train_norm, X_val_norm, y_train, y_val, means, stds


def stratified_split_and_normalize(
    df: pd.DataFrame, 
    target_column: str, 
    exclude_cols: Optional[List[str]] = None, 
    test_ratio: float = 0.2, 
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Performs stratified split, handles missing values in numeric columns using mean 
    imputation, and normalizes features based on the training set.

    Assumes input DataFrame primarily contains numerical features after target separation.

    Args:
        df (pd.DataFrame): DataFrame to split, impute, and normalize.
        target_column (str): Name of the target column.
        exclude_cols (Optional[List[str]], optional): Columns to exclude from normalization. 
            Defaults to None.
        test_ratio (float, optional): Proportion of data for the test set. Defaults to 0.2.
        random_state (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]: 
        A tuple containing: X_train_norm, X_test_norm, y_train, y_test, means, stds.
    """
    # 1. Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column].values
    
    # 2. Split the data (stratified)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_ratio, random_state)
    
    # 3. Handle missing values (mean imputation for numeric columns)
    X_train = handle_missing_values(X_train, train_df=None) # Impute train based on itself
    X_test = handle_missing_values(X_test, train_df=X_train) # Impute test based on train mean

    # 4. Normalize - fit on train, transform both
    if exclude_cols is None:
        exclude_cols_norm = []
    else:
        exclude_cols_norm = exclude_cols.copy()
        
    X_train_norm, means, stds = normalize(X_train, exclude_cols=exclude_cols_norm)
    X_test_norm, _, _ = normalize(X_test, means=means, stds=stds, exclude_cols=exclude_cols_norm)
    
    return X_train_norm, X_test_norm, y_train, y_test, means, stds

def create_stratified_k_folds(
    X: Union[pd.DataFrame, np.ndarray], # Accept DataFrame or ndarray for X
    y: np.ndarray, 
    k: int = 5, 
    random_state: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates indices for K-Fold cross-validation with stratification.

    Ensures that the proportion of samples for each class is approximately 
    the same across all folds as in the original dataset.

    Args:
        X (Union[pd.DataFrame, np.ndarray]): Feature data. Only its length is used, 
            but passed for API consistency.
        y (np.ndarray): Array of target labels.
        k (int, optional): The number of folds. Must be at least 2. Defaults to 5.
        random_state (int, optional): Seed for the random number generator for 
            reproducible fold assignments. Defaults to None.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of length `k`. Each element is a tuple 
            containing two NumPy arrays: (train_indices, validation_indices) for that fold.
            
    Raises:
        ValueError: If k is less than 2 or greater than the number of samples 
                    in the smallest class.
    """
    if k < 2:
        raise ValueError("Number of folds k must be at least 2.")
        
    if random_state is not None:
        np.random.seed(random_state)

    y_arr = np.asarray(y)
    n_samples = len(y_arr)
    indices = np.arange(n_samples)
    
    # Get unique classes and their counts
    unique_labels, y_inversed = np.unique(y_arr, return_inverse=True)
    class_counts = np.bincount(y_inversed)
    
    # Check if k is feasible given the smallest class size
    min_class_size = np.min(class_counts)
    if k > min_class_size:
        raise ValueError(f"Cannot create {k} folds with stratification. The smallest "
                         f"class has only {min_class_size} samples. Reduce k or handle "
                         f"small classes.")

    # Initialize list to hold validation indices for each fold
    # Stratify by distributing indices of each class across folds
    per_fold_indices: List[List[int]] = [[] for _ in range(k)]
    
    for class_label_idx, count in enumerate(class_counts):
        # Get indices for the current class
        class_indices_original = indices[y_inversed == class_label_idx]
        # Shuffle class indices
        np.random.shuffle(class_indices_original)
        
        # Distribute shuffled indices cyclically among folds
        for i, idx in enumerate(class_indices_original):
            target_fold = i % k
            per_fold_indices[target_fold].append(idx)

    # Create the final train/validation splits
    fold_splits: List[Tuple[np.ndarray, np.ndarray]] = []
    all_indices_set = set(indices)
    
    for i in range(k):
        # Validation indices for fold i are those assigned above
        val_indices = np.array(per_fold_indices[i], dtype=int)
        
        # Training indices are all indices NOT in the validation set for this fold
        val_indices_set = set(val_indices)
        train_indices = np.array(list(all_indices_set - val_indices_set), dtype=int)
        
        # Sort indices for potential caching benefits (optional)
        # train_indices.sort()
        # val_indices.sort()
        
        fold_splits.append((train_indices, val_indices))

    return fold_splits
