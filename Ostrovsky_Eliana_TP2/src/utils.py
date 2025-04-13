# src/utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from typing import List, Dict, Optional, Union, Tuple, Any, Sequence

# Type Definitions for clarity
ArrayLike = Union[List[Any], np.ndarray, pd.Series]
Model = Any # Placeholder for generic model type
Numeric = Union[int, float]
ClassWeights = Optional[Union[Dict[Any, float], str]] # For class weights
Labels = Optional[Sequence[Any]] 
TargetNames = Optional[List[str]]

def print_class_balance_report(
    y: ArrayLike, 
    class_names: TargetNames = None
) -> Dict[str, Any]:
    """
    Calculates and prints a report on the balance of classes in the target variable.

    Args:
        y (ArrayLike): Array or Series containing the target class labels.
        class_names (TargetNames, optional): List of names corresponding to the unique 
            classes found in `y`. If None, class labels are used directly as names (converted 
            to string). Length must match the number of unique classes. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing the calculated report details:
            - 'counts' (Dict[str, int]): Count of samples per class name.
            - 'proportions' (Dict[str, float]): Proportion of samples per class name.
            - 'total_samples' (int): Total number of samples.
            
    Raises:
        ValueError: If `class_names` is provided and its length doesn't match the 
                    number of unique classes in `y`.
    """
    y_arr = np.asarray(y)
    # Ensure consistent type for finding unique labels, convert to string
    y_str = y_arr.astype(str) 
    
    unique_classes, counts = np.unique(y_str, return_counts=True)
    total_samples = len(y_arr)
    
    # Determine class names to use
    if class_names is None:
        effective_class_names = list(unique_classes) # Use the string versions found
    else:
        if len(class_names) != len(unique_classes):
            raise ValueError(f"Length of class_names ({len(class_names)}) must match the "
                             f"number of unique classes found in y ({len(unique_classes)}).")
        effective_class_names = list(class_names) # Use provided names
        
    # Map unique string labels to effective names (handles case where labels != names)
    class_map = {cls_str: name for cls_str, name in zip(unique_classes, effective_class_names)}

    # Build the report dictionary
    report: Dict[str, Any] = {
        'counts': {},
        'proportions': {},
        'total_samples': total_samples
    }
    for cls_str, count in zip(unique_classes, counts):
        name = class_map[cls_str]
        proportion = count / total_samples if total_samples > 0 else 0.0
        report['counts'][name] = count
        report['proportions'][name] = proportion

    # Print the formatted report
    print("\nClass Balance Report")
    print("=" * 20)
    print(f"Total samples: {total_samples}")
    print("\nCounts per class:")
    # Sort items for consistent print order
    for name, count in sorted(report['counts'].items()):
        proportion = report['proportions'][name]
        print(f"- {name}: {count} samples ({proportion:.2%})")
    print("-" * 20)

    return report


def calculate_sample_weights(
    y: ArrayLike, 
    class_weights: ClassWeights = 'balanced'
) -> np.ndarray:
    """
    Calculates sample weights based on class distribution.

    Useful for handling imbalanced datasets during model training by giving 
    more weight to samples from under-represented classes.

    Args:
        y (ArrayLike): Array or Series containing the target class labels.
        class_weights (ClassWeights, optional): Strategy for weights:
            - 'balanced': Automatically compute weights inversely proportional 
                          to class frequencies (n_samples / (n_classes * n_samples_class)).
            - dict: A dictionary mapping class labels to specific weight values 
                    (e.g., {0: 0.8, 1: 1.2}).
            - None: No weights are applied (returns array of ones).
            Defaults to 'balanced'.

    Returns:
        np.ndarray: An array of sample weights, one for each sample in `y`.
        
    Raises:
        ValueError: If `class_weights` is a dict but contains labels not found in `y`.
    """
    y_arr = np.asarray(y)
    n_samples = len(y_arr)

    if class_weights is None:
        return np.ones(n_samples, dtype=float) # Return uniform weights

    elif isinstance(class_weights, dict):
        # Apply weights directly from the provided dictionary
        try:
            # Use np.vectorize or list comprehension for mapping
            # Need to handle labels in dict not matching y values
            unique_y = np.unique(y_arr)
            if not all(cls in class_weights for cls in unique_y):
                 missing = [cls for cls in unique_y if cls not in class_weights]
                 print(f"Warning: class_weights dict missing weights for labels: {missing}. "
                       f"Assigning weight 1.0 to these.")
            
            # Use get with default 1.0 for missing keys
            sample_weights = np.array([class_weights.get(cls, 1.0) for cls in y_arr], dtype=float)
            return sample_weights
        except KeyError as e:
             # This path might be less likely now with .get()
             raise ValueError(f"Label {e} found in y but not in class_weights dictionary.") from e
             
    elif class_weights == 'balanced':
        # Calculate balanced weights
        unique_classes, counts = np.unique(y_arr, return_counts=True)
        n_classes = len(unique_classes)
        
        if n_classes < 2:
            # print("Warning: Cannot compute balanced weights with fewer than 2 classes. Returning uniform weights.")
            return np.ones(n_samples, dtype=float)
        
        # Calculate weights: n_samples / (n_classes * count_per_class)
        # Handle potential division by zero if a class count is 0 (shouldn't happen with unique)
        weights_per_class = n_samples / (n_classes * counts)
        
        # Create a mapping from class label to its calculated weight
        weight_map = {cls: weight for cls, weight in zip(unique_classes, weights_per_class)}
        
        # Map weights back to each sample
        sample_weights = np.array([weight_map[cls] for cls in y_arr], dtype=float)
        return sample_weights
        
    else:
        raise ValueError(f"Invalid class_weights value: {class_weights}. "
                         "Expected 'balanced', None, or a dictionary.")

def save_model(model: Any, filename: str) -> None:
    """
    Saves a trained model object to a file using pickle.

    Args:
        model (Any): The model object to save (e.g., a classifier instance).
        filename (str): The path and name of the file to save the model to 
                        (conventionally ending in .pkl or .joblib).
                        
    Raises:
        IOError: If there's an error writing the file.
        pickle.PicklingError: If the model object cannot be pickled.
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model successfully saved to {filename}")
    except (IOError, pickle.PicklingError) as e:
        print(f"Error saving model to {filename}: {e}")
        raise # Re-raise the exception


def load_model(filename: str) -> Any:
    """
    Loads a model object from a file saved using pickle.

    Args:
        filename (str): The path and name of the file containing the saved model.

    Returns:
        Any: The loaded model object.

    Raises:
        IOError: If the file cannot be opened or read.
        pickle.UnpicklingError: If the file content is not a valid pickle stream 
                                or if there are issues deserializing the object 
                                (e.g., missing class definitions).
    """
    try:
        with open(filename, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"Model successfully loaded from {filename}")
        return loaded_model
    except (IOError, pickle.UnpicklingError) as e:
        print(f"Error loading model from {filename}: {e}")
        raise # Re-raise the exception