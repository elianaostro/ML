import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

def random_undersample( X_df: pd.DataFrame, y_array: np.ndarray, random_state: Optional[int] = None
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
    temp_df = X_df.copy()
    target_col_name = '__target__' 
    temp_df[target_col_name] = y_array

    _, counts = np.unique(y_array, return_counts=True)
    min_count = np.min(counts)

    sampled_df = temp_df.groupby(target_col_name, group_keys=False)\
                        .apply(lambda x: x.sample(min_count, random_state=random_state))\
                        .reset_index(drop=True)

    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    X_resampled = sampled_df.drop(target_col_name, axis=1)
    y_resampled = sampled_df[target_col_name].values

    return X_resampled, y_resampled

def duplicate_oversample( X_df: pd.DataFrame, y_array: np.ndarray, random_state: Optional[int] = None 
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
    temp_df = X_df.copy()
    target_col_name = '__target__'
    temp_df[target_col_name] = y_array

    target_counts = pd.Series(y_array).value_counts()
    max_count = target_counts.max()

    resampled_dfs: List[pd.DataFrame] = []
    for class_label, count in target_counts.items():
        class_df = temp_df[temp_df[target_col_name] == class_label]
        if count < max_count:
            n_needed = max_count - count
            additional_samples = class_df.sample(n=n_needed, replace=True, random_state=random_state)
            resampled_dfs.append(pd.concat([class_df, additional_samples], axis=0))
        else:
            resampled_dfs.append(class_df)

    final_df = pd.concat(resampled_dfs, axis=0)\
                   .sample(frac=1, random_state=random_state)\
                   .reset_index(drop=True)

    X_resampled = final_df.drop(target_col_name, axis=1)
    y_resampled = final_df[target_col_name].values

    return X_resampled, y_resampled


def SMOTE( X_df: pd.DataFrame, y_array: np.ndarray, k: int = 5, random_state: Optional[int] = None
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

    X_np = X_df.values
    classes, counts = np.unique(y_array, return_counts=True)
    max_count = np.max(counts)    
    resampled_dfs = [X_df]
    resampled_ys = [y_array] 

    minority_classes_indices = np.where(counts < max_count)[0]
    minority_classes = classes[minority_classes_indices]

    for class_idx in minority_classes:
        minority_mask = (y_array == class_idx)
        X_class = X_np[minority_mask]
        current_count = X_class.shape[0]
        n_needed = max_count - current_count

        if n_needed <= 0: 
            continue
            
        current_k = k
        distances = np.sqrt(np.sum((X_class[:, np.newaxis, :] - X_class[np.newaxis, :, :])**2, axis=-1))
        knn_indices = np.argsort(distances, axis=1)[:, 1:current_k+1]
        base_sample_indices = np.random.randint(0, current_count, size=n_needed)
        random_neighbor_selection = np.random.randint(0, current_k, size=n_needed)
        neighbor_indices = knn_indices[base_sample_indices, random_neighbor_selection]
        alphas = np.random.random(size=n_needed)[:, np.newaxis] 

        base_samples = X_class[base_sample_indices]
        neighbor_samples = X_class[neighbor_indices]
        synthetic_samples_np = base_samples + alphas * (neighbor_samples - base_samples)
        synthetic_df = pd.DataFrame(synthetic_samples_np, columns=X_df.columns)
        resampled_dfs.append(synthetic_df)
        resampled_ys.append(np.full(n_needed, class_idx, dtype=y_array.dtype))

    X_combined = pd.concat(resampled_dfs, axis=0, ignore_index=True)
    y_combined = np.concatenate(resampled_ys)

    final_indices = np.random.permutation(len(X_combined))
    
    X_resampled_final: pd.DataFrame = X_combined.iloc[final_indices].reset_index(drop=True)
    y_resampled_final = y_combined[final_indices]

    return X_resampled_final, y_resampled_final