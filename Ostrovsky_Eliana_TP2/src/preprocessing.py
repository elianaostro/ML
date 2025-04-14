from math import dist
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
from models import KMeans
from scipy.spatial.distance import cdist

def apply_variable_limits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies predefined limits to specific columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with values outside the limits set to NaN.
    """
    df_limited = df.copy()
    column_ranges = {
        'CellSize': (0, 1000 - 0.001),
        'NucleusDensity': (0, 50 - 0.001),
        'CellAdhesion': (0, 1),
        'NuclearMembrane': (1, 5),
        'OxygenSaturation': (0, 100),
        'Vascularization': (0, 10),
        'InflammationMarkers': (0, 100)
    }
    for column, (min_val, max_val) in column_ranges.items():
        df_limited[column] = np.where(
            df_limited[column].between(min_val, max_val, inclusive='both') | df_limited[column].isna(),
            df_limited[column],
            df_limited[column]/10
        )
    return df_limited

def remove_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sets negative numeric values (excluding binary columns) in the DataFrame to NaN.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with negative numeric values set to NaN.
    """
    df_non_negative = df.copy()
    numeric_cols = df_non_negative.select_dtypes(include=np.number).columns.difference(['CellAdhesion', 'NuclearMembrane', 'OxygenSaturation', 'Vascularization', 'InflammationMarkers'])
    numeric_cols = [col for col in numeric_cols if not set(df_non_negative[col].dropna().unique()).issubset({0, 1})]
    df_numeric = df_non_negative[numeric_cols]
    df_non_negative[numeric_cols] = df_numeric.where(df_numeric >= 0, np.nan)
    return df_non_negative

def remove_outliers_iqr(df: pd.DataFrame, underlimit: float = 0.15, uperlimit: float = 0.85) -> pd.DataFrame:
    """
    Removes outliers from numeric columns (excluding specified and binary-like)
    using the IQR method based on specified percentiles. Outliers are set to NaN.

    Args:
        df (pd.DataFrame): The input DataFrame.
        underlimit (float): Lower percentile for IQR calculation. Defaults to 0.15.
        uperlimit (float): Upper percentile for IQR calculation. Defaults to 0.85.

    Returns:
        pd.DataFrame: DataFrame with outliers set to NaN.
    """
    df_no_outliers = df.copy()
    numeric_cols = df_no_outliers.select_dtypes(include=np.number).columns.difference(['CellAdhesion', 'NuclearMembrane', 'OxygenSaturation', 'Vascularization', 'InflammationMarkers'])
    numeric_cols = [col for col in numeric_cols if not set(df_no_outliers[col].dropna().unique()).issubset({0, 1})]

    q_lower = df_no_outliers[numeric_cols].quantile(underlimit)
    q_upper = df_no_outliers[numeric_cols].quantile(uperlimit)
    iqr = q_upper - q_lower

    lower_bound = q_lower - 1.5 * iqr
    upper_bound = q_upper + 1.5 * iqr

    for column in numeric_cols:
        col_data = df_no_outliers[column]
        is_outlier = (col_data < lower_bound[column]) | (col_data > upper_bound[column])
        df_no_outliers[column] = col_data.where(~is_outlier, np.nan)
    return df_no_outliers

def remove_outliers(df: pd.DataFrame, underlimit: float = 0.15, uperlimit: float = 0.85) -> pd.DataFrame:
    """
    Identifies and removes outliers from within each cluster (with 2 clusters)
    for each numeric feature (excluding specified and binary-like) using a custom
    K-Means function. Outliers within each cluster are identified using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        underlimit (float): Lower percentile for IQR calculation. Defaults to 0.15.
        uperlimit (float): Upper percentile for IQR calculation. Defaults to 0.85.

    Returns:
        pd.DataFrame: DataFrame with outliers within each cluster set to NaN.
    """
    df_no_outliers = df.copy()
    numeric_cols = df_no_outliers.select_dtypes(include=np.number).columns.difference(['CellAdhesion', 'NuclearMembrane', 'OxygenSaturation', 'Vascularization', 'InflammationMarkers'])
    numeric_cols = [col for col in numeric_cols if not set(df_no_outliers[col].dropna().unique()).issubset({0, 1})]

    iqr_multiplier = 1.5  

    for column in numeric_cols:
        data_col_df = df_no_outliers[[column]].dropna()
        if len(data_col_df) < 2: 
            continue

        try:
            labels, _ = KMeans(data_col_df, n_clusters=2, random_state=42)
            clusters = pd.Series(labels, index=data_col_df.index)
        except ValueError as e:
            print(f"Error during custom KMeans for column {column}: {e}")
            continue

        for cluster_label in np.unique(clusters):
            cluster_data = data_col_df[clusters == cluster_label]

            if len(cluster_data) > 2:  
                Q1 = cluster_data[column].quantile(underlimit)
                Q3 = cluster_data[column].quantile(uperlimit)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR

                outlier_indices_cluster = cluster_data.index[(cluster_data[column] < lower_bound) | (cluster_data[column] > upper_bound)]

                df_no_outliers.loc[outlier_indices_cluster, column] = np.nan

    return df_no_outliers

def remove_high_nan_rows(df: pd.DataFrame, nan_threshold: int = 7) -> pd.DataFrame:
    """
    Removes rows from the DataFrame that have a number of NaN values greater than or equal to the specified threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        nan_threshold (int): The maximum number of NaN values allowed per row.
                             Rows with this many or more NaNs will be removed. Defaults to 7.

    Returns:
        pd.DataFrame: DataFrame with rows containing too many NaN values removed.
    """
    df_low_nan = df.copy()
    df_low_nan['NaN_Count'] = df_low_nan.isna().sum(axis=1)
    df_low_nan = df_low_nan[df_low_nan['NaN_Count'] < nan_threshold]
    df_low_nan = df_low_nan.drop(columns=['NaN_Count'])
    return df_low_nan

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


# def handle_missing_values(df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
#     """
#     Handles missing values (NaN) in numerical columns of a DataFrame using mean imputation.

#     Assumes the input DataFrame contains only columns to be treated numerically 
#     for the purpose of imputation.

#     Args:
#         df (pd.DataFrame): DataFrame with potential missing values in numerical columns.
#         train_df (Optional[pd.DataFrame], optional): Reference DataFrame (e.g., training set) 
#             to calculate means from, ensuring consistency. If None, means are 
#             calculated from the input `df` itself. Defaults to None.

#     Returns:
#         pd.DataFrame: DataFrame with missing values in numerical columns imputed using the mean.
#     """
#     df_imputed = df.copy()
    
#     reference_df = train_df if train_df is not None else df_imputed

#     numeric_cols = df_imputed.select_dtypes(include=np.number).columns
    
#     if not numeric_cols.empty:
#         for col in numeric_cols:
#             if df_imputed[col].isnull().any():
#                 fill_val = reference_df[col].mean()
#                 if pd.isna(fill_val):
#                     fill_val = 0.0
#                     print(f"Warning: Mean for column '{col}' could not be calculated (all NaNs?). Imputing with 0.0.")
#                 df_imputed[col] = df_imputed[col].fillna(fill_val)

#     return df_imputed

def handle_missing_values(df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Imputes missing values in a DataFrame using a simplified k-Nearest Neighbors approach.

    For each column with missing values, it uses other numerical columns from the
    `reference_df` (or `df` itself if `reference_df` is None) to find the
    k-nearest neighbors of the rows with missing values in `df`, and imputes
    with the mean of the neighbors' values in that column from `reference_df`.

    Args:
        df (pd.DataFrame): DataFrame with potential missing values to be imputed.
        reference_df (Optional[pd.DataFrame], optional): DataFrame to use as a
            reference for finding nearest neighbors. If None, `df` is used.
            Defaults to None.
        n_neighbors (int, optional): The number of nearest neighbors to consider
            for imputation. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed using k-NN.
    """
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=np.number).columns

    if train_df is None:
        train_df = df_imputed

    ref_numeric_cols = train_df.select_dtypes(include=np.number).columns
    common_numeric_cols = list(set(numeric_cols) & set(ref_numeric_cols))

    for col_to_impute in numeric_cols:
        if df_imputed[col_to_impute].isnull().any():
            missing_mask = df_imputed[col_to_impute].isnull()
            missing_data = df_imputed[missing_mask][common_numeric_cols].values
            missing_indices = df_imputed[missing_mask].index

            observed_ref_data = train_df[train_df[col_to_impute].notna()][common_numeric_cols].values
            observed_ref_indices = train_df[train_df[col_to_impute].notna()].index

            if observed_ref_data.shape[0] >= n_neighbors and missing_data.shape[0] > 0:
                distances = cdist(missing_data, observed_ref_data, metric='euclidean')
                nearest_neighbor_indices = np.argsort(distances, axis=1)[:, :n_neighbors]

                for i, missing_row_idx in enumerate(missing_indices):
                    neighbor_values = train_df.loc[observed_ref_indices[nearest_neighbor_indices[i]], col_to_impute].values
                    imputed_value = np.nanmean(neighbor_values)
                    df_imputed.loc[missing_row_idx, col_to_impute] = imputed_value

            elif missing_data.shape[0] > 0:
                fill_value = train_df[col_to_impute].mean()
                if pd.isna(fill_value):
                    fill_value = 0.0
                df_imputed.loc[missing_indices, col_to_impute] = fill_value

    return df_imputed

def clean_data(df: pd.DataFrame, underlimit: float = 0.15, uperlimit: float = 0.85, nan_threshold: int = 7) -> pd.DataFrame:
    """
    Applies data cleaning steps by calling individual cleaning functions.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        underlimit (float): Lower percentile for IQR-based outlier detection. Defaults to 0.15.
        uperlimit (float): Upper percentile for IQR-based outlier detection. Defaults to 0.85.
        nan_threshold (int): The maximum number of NaN values allowed per row. Defaults to 7.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.copy()
    df_cleaned = apply_variable_limits(df_cleaned)
    df_cleaned = remove_negative_values(df_cleaned)
    df_cleaned = remove_outliers_iqr(df_cleaned, underlimit, uperlimit)
    df_cleaned = remove_high_nan_rows(df_cleaned, nan_threshold)
    return df_cleaned

def preprocess_data(df: pd.DataFrame, underlimit: float = 0.15, uperlimit: float = 0.85, train_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Preprocesses the DataFrame by applying cleaning and categorical handling.

    This includes:
    - Cleaning the DataFrame (clipping, NaN handling, outlier detection).
    - Handling categorical features (one-hot encoding, binary conversion).

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.
        underlimit (float): Lower percentile for IQR-based outlier detection. Defaults to 0.15.
        uperlimit (float): Upper percentile for IQR-based outlier detection. Defaults to 0.85.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df_processed = clean_data(df, underlimit=underlimit, uperlimit=uperlimit)
    df_processed = handle_categorical_features(df_processed)
    df_processed = handle_missing_values(df_processed, train_df=train_df)
    
    return df_processed
    
def normalize( X: pd.DataFrame, means: Optional[pd.Series] = None, stds: Optional[pd.Series] = None, exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
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


def split_data( df: pd.DataFrame, target_column: str, train_ratio: float = 0.8, random_state: Optional[int] = None
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

def stratified_split( df: pd.DataFrame, target_column: str, test_ratio: float = 0.2, random_state: Optional[int] = None
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Performs a stratified split of a DataFrame into training and test sets.

    Maintains the same proportion of classes in both the train and test sets
    as in the original dataset.

    Args:
        df (pd.DataFrame): The complete DataFrame containing both features
                           and the target column.
        target_column (str): The name of the column in 'df' that contains the
                             labels or target variable.
        test_ratio (float, optional): Proportion of the dataset to include in the
                                      test split. Defaults to 0.2.
        random_state (Optional[int], optional): Seed for reproducibility.
                                               Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        A tuple containing: X_train, X_test, y_train, y_test.
        X_train and X_test are DataFrames, y_train and y_test are NumPy arrays.

    Raises:
        ValueError: If 'target_column' is not found in 'df'.
        ValueError: If 'test_ratio' is not between 0 and 1 (exclusive).
    """

    y = df[target_column].to_numpy()
    X = df.drop(columns=[target_column])

    if random_state is not None:
        np.random.seed(random_state)

    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    class_indices = { i: np.where(y_indices == i)[0] for i in range(n_classes)}

    train_indices = []
    test_indices = []

    for i in range(n_classes):
        indices_for_class = class_indices[i]
        n_class_samples = len(indices_for_class)

        n_test_class = int(np.round(n_class_samples * test_ratio))
        if test_ratio > 0 and n_test_class == 0 and n_class_samples > 0:
             n_test_class = 1

        n_test_class = min(n_test_class, n_class_samples)

        if n_class_samples > 1 and n_test_class == n_class_samples:
            n_test_class -= 1

        n_train_class = n_class_samples - n_test_class

        np.random.shuffle(indices_for_class)

        train_indices.extend(indices_for_class[:n_train_class])
        test_indices.extend(indices_for_class[n_train_class:])

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def split_and_normalize( df: pd.DataFrame, target_column: str, exclude_cols: Optional[List[str]] = [], train_ratio: float = 0.8, random_state: Optional[int] = None, stratified: bool = False
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Splits data into train/validation sets, handles missing values in numeric columns 
    using mean imputation, and normalizes features based on the training set.

    Supports both stratified and non-stratified splitting.

    Assumes input DataFrame primarily contains numerical features after target separation.

    Args:
        df (pd.DataFrame): DataFrame to split, impute, and normalize.
        target_column (str): Name of the target column.
        exclude_cols (Optional[List[str]], optional): Columns to exclude from normalization. 
            Defaults to an empty list.
        train_ratio (float, optional): Proportion of data for training. Defaults to 0.8.
        random_state (int, optional): Seed for reproducibility. Defaults to None.
        stratified (bool, optional): Whether to perform stratified splitting. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series]: 
        A tuple containing: X_train_norm, X_val_norm, y_train, y_val, means, stds.
    """
    if stratified:
        X_train, X_val, y_train, y_val = stratified_split(df, target_column, test_ratio=1 - train_ratio, random_state=random_state)
    else:
        X_train, X_val, y_train, y_val = split_data(df, target_column, train_ratio, random_state)
    
    X_train = handle_missing_values(X_train, train_df=None) 
    X_val = handle_missing_values(X_val, train_df=X_train)  

    X_train_norm, means, stds = normalize(X_train, exclude_cols=exclude_cols)
    X_val_norm, _, _ = normalize(X_val, means=means, stds=stds, exclude_cols=exclude_cols)
    
    return X_train_norm, X_val_norm, y_train, y_val

def create_stratified_k_folds(X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, k: int = 5, random_state: Optional[int] = None
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
    
    unique_labels, y_inversed = np.unique(y_arr, return_inverse=True)
    class_counts = np.bincount(y_inversed)
    
    min_class_size = np.min(class_counts)
    if k > min_class_size:
        raise ValueError(f"Cannot create {k} folds with stratification. The smallest "
                         f"class has only {min_class_size} samples. Reduce k or handle "
                         f"small classes.")

    per_fold_indices: List[List[int]] = [[] for _ in range(k)]
    
    for class_label_idx, count in enumerate(class_counts):
        class_indices_original = indices[y_inversed == class_label_idx]
        np.random.shuffle(class_indices_original)
        
        for i, idx in enumerate(class_indices_original):
            target_fold = i % k
            per_fold_indices[target_fold].append(idx)

    fold_splits: List[Tuple[np.ndarray, np.ndarray]] = []
    all_indices_set = set(indices)
    
    for i in range(k):
        val_indices = np.array(per_fold_indices[i], dtype=int)
        
        val_indices_set = set(val_indices)
        train_indices = np.array(list(all_indices_set - val_indices_set), dtype=int)
        
        fold_splits.append((train_indices, val_indices))

    return fold_splits
