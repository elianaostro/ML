import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union, Any

def random_undersample(
    X_df: pd.DataFrame, 
    y_array: np.ndarray, 
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Perform random undersampling to balance the dataset.

    This function reduces the number of samples in the majority class(es)
    by randomly selecting samples until all classes have the same number 
    of samples as the original minority class.

    Args:
        X_df (pd.DataFrame): DataFrame containing the features.
        y_array (np.ndarray): NumPy array containing the target labels.
        random_state (Optional[int], optional): Seed for the random number 
            generator for reproducibility. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A tuple containing the resampled 
            features DataFrame and the corresponding resampled target labels array.
            The data is shuffled.
    """
    # Create a temporary DataFrame including the target for easier grouping
    temp_df = X_df.copy()
    target_col_name = '__target__' # Use a temporary, unlikely column name
    temp_df[target_col_name] = y_array

    # Determine the number of samples in the smallest class
    _, counts = np.unique(y_array, return_counts=True)
    min_count = np.min(counts)

    # Group by target, sample min_count from each group, and recombine
    # Use reset_index() to handle potential MultiIndex after apply
    sampled_df = temp_df.groupby(target_col_name, group_keys=False)\
                        .apply(lambda x: x.sample(min_count, random_state=random_state))\
                        .reset_index(drop=True)

    # Shuffle the final dataset to avoid order by class
    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Separate features and target
    X_resampled = sampled_df.drop(target_col_name, axis=1)
    y_resampled = sampled_df[target_col_name].values

    return X_resampled, y_resampled

def duplicate_oversample(
    X_df: pd.DataFrame, 
    y_array: np.ndarray, 
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Perform oversampling by duplicating samples from minority classes.

    This function increases the number of samples in the minority class(es)
    by randomly duplicating existing samples until all classes have the same 
    number of samples as the original majority class.

    Args:
        X_df (pd.DataFrame): DataFrame containing the features.
        y_array (np.ndarray): NumPy array containing the target labels.
        random_state (Optional[int], optional): Seed for the random number 
            generator for reproducibility. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A tuple containing the resampled 
            features DataFrame and the corresponding resampled target labels array.
            The data is shuffled.
    """
    # Create a temporary DataFrame including the target
    temp_df = X_df.copy()
    target_col_name = '__target__'
    temp_df[target_col_name] = y_array

    # Determine the number of samples in the largest class
    target_counts = pd.Series(y_array).value_counts()
    max_count = target_counts.max()

    resampled_dfs: List[pd.DataFrame] = []
    # Iterate through each class label and its count
    for class_label, count in target_counts.items():
        class_df = temp_df[temp_df[target_col_name] == class_label]
        # If the class count is less than the max count, oversample it
        if count < max_count:
            n_needed = max_count - count
            # Sample with replacement to duplicate instances
            additional_samples = class_df.sample(n=n_needed, replace=True, random_state=random_state)
            # Combine original and duplicated samples for this class
            resampled_dfs.append(pd.concat([class_df, additional_samples], axis=0))
        else:
            # If it's a majority class, just add its original samples
            resampled_dfs.append(class_df)

    # Combine all class DataFrames and shuffle the result
    final_df = pd.concat(resampled_dfs, axis=0)\
                   .sample(frac=1, random_state=random_state)\
                   .reset_index(drop=True)

    # Separate features and target
    X_resampled = final_df.drop(target_col_name, axis=1)
    y_resampled = final_df[target_col_name].values

    return X_resampled, y_resampled


def SMOTE(
    X_df: pd.DataFrame, 
    y_array: np.ndarray, 
    k: int = 5, 
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Perform SMOTE (Synthetic Minority Over-sampling TEchnique) using a 
    vectorized approach.

    This function generates synthetic samples for the minority class(es) based 
    on their k-nearest neighbors until all classes have the same number of 
    samples as the original majority class.

    Args:
        X_df (pd.DataFrame): DataFrame containing the features.
        y_array (np.ndarray): NumPy array containing the target labels.
        k (int, optional): Number of nearest neighbors to consider for 
            generating synthetic samples. Defaults to 5.
        random_state (Optional[int], optional): Seed for the random number 
            generator for reproducibility. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A tuple containing the resampled 
            (original + synthetic) features DataFrame and the corresponding 
            target labels array. The data is shuffled.
            
    Notes:
        - If a minority class has fewer than k+1 samples, k will be adjusted
          downwards for that class to avoid errors.
        - Assumes numerical features for distance calculations.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Convert features DataFrame to NumPy array for efficient computation
    X_np: np.ndarray = X_df.values
    # Get unique classes and their counts
    classes: np.ndarray
    counts: np.ndarray
    classes, counts = np.unique(y_array, return_counts=True)
    
    # Determine the target size (number of samples in the majority class)
    max_count: int = np.max(counts)
    
    # Lists to store original and synthetic data components
    resampled_dfs: List[pd.DataFrame] = [X_df] # Start with original features
    resampled_ys: List[np.ndarray] = [y_array] # Start with original labels

    # Identify minority classes (those needing oversampling)
    minority_classes_indices: np.ndarray = np.where(counts < max_count)[0]
    minority_classes: np.ndarray = classes[minority_classes_indices]

    # Iterate through each minority class to generate synthetic samples
    for class_idx in minority_classes:
        # Get samples belonging to the current minority class
        minority_mask: np.ndarray = (y_array == class_idx)
        X_class: np.ndarray = X_np[minority_mask]
        current_count: int = X_class.shape[0]
        n_needed: int = max_count - current_count

        # Skip if no samples are needed for this class
        if n_needed <= 0: 
            continue
            
        # Adjust k if there are not enough neighbors
        current_k: int = k
        if current_count <= k:
             # Adjust k to be the number of available neighbors (excluding self)
             current_k = max(1, current_count - 1) 
             print(f"Warning: Not enough samples ({current_count}) for k={k} "
                   f"in class {class_idx}. Using k={current_k}.")
        
        # --- Efficient KNN Calculation ---
        # Calculate pairwise Euclidean distances within the class samples
        # Using broadcasting for efficiency: (n, 1, d) - (1, n, d) -> (n, n, d) -> sum(axis=-1) -> sqrt
        distances: np.ndarray = np.sqrt(np.sum((X_class[:, np.newaxis, :] - X_class[np.newaxis, :, :])**2, axis=-1))
        
        # Find indices of the k nearest neighbors for each sample
        # argsort returns indices that would sort the array; take indices 1 to k+1 to exclude self
        knn_indices: np.ndarray = np.argsort(distances, axis=1)[:, 1:current_k+1]
        # --- End KNN Calculation ---

        # --- Vectorized Synthetic Sample Generation ---
        # 1. Choose 'n_needed' random base samples from the minority class (with replacement)
        base_sample_indices: np.ndarray = np.random.randint(0, current_count, size=n_needed)

        # 2. Choose a random neighbor for each base sample from its k neighbors
        #    Select a random column index from the knn_indices for each base sample
        random_neighbor_selection: np.ndarray = np.random.randint(0, current_k, size=n_needed)
        neighbor_indices: np.ndarray = knn_indices[base_sample_indices, random_neighbor_selection]

        # 3. Generate 'n_needed' random interpolation factors (alphas) between 0 and 1
        #    Reshape to (n_needed, 1) for broadcasting
        alphas: np.ndarray = np.random.random(size=n_needed)[:, np.newaxis] 

        # 4. Calculate synthetic samples: base + alpha * (neighbor - base)
        base_samples: np.ndarray = X_class[base_sample_indices]
        neighbor_samples: np.ndarray = X_class[neighbor_indices]
        synthetic_samples_np: np.ndarray = base_samples + alphas * (neighbor_samples - base_samples)
        # --- End Vectorized Generation ---

        # Convert synthetic samples back to a DataFrame, preserving column names
        synthetic_df = pd.DataFrame(synthetic_samples_np, columns=X_df.columns)

        # Append synthetic data to the lists
        resampled_dfs.append(synthetic_df)
        # Append the corresponding labels for the synthetic samples
        resampled_ys.append(np.full(n_needed, class_idx, dtype=y_array.dtype)) # Match original dtype

    # Combine original and all synthetic data
    X_combined: pd.DataFrame = pd.concat(resampled_dfs, axis=0, ignore_index=True)
    y_combined: np.ndarray = np.concatenate(resampled_ys)

    # Shuffle the combined dataset
    final_indices: np.ndarray = np.random.permutation(len(X_combined))
    
    # Use .iloc for shuffling DataFrames based on indices and reset index
    X_resampled_final: pd.DataFrame = X_combined.iloc[final_indices].reset_index(drop=True)
    y_resampled_final: np.ndarray = y_combined[final_indices]

    return X_resampled_final, y_resampled_final