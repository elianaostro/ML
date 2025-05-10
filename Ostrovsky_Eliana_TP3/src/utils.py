# src/utils.py
import sys
from typing import Dict, Any
import pickle
import os

def update_progress_bar(current, total, bar_length=50, metrics=None):
    """
    Display a progress bar with optional metrics.
    
    Args:
        current: Current progress value
        total: Total value
        bar_length: Length of the progress bar
        metrics: Dictionary of metrics to display
    """
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    metrics_str = ""
    if metrics:
        metrics_str = " - " + " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    sys.stdout.write(f"\rEpoch: [{arrow + spaces}] {int(percent * 100)}%{metrics_str}")
    sys.stdout.flush()

def save_model(model: Any, filename: str) -> None:
    """
    Saves a trained model object to a file inside the 'modelos' directory using pickle.

    Args:
        model (Any): The model object to save (e.g., a classifier instance).
        filename (str): The name of the file to save the model to 
                        (conventionally ending in .pkl or .joblib).
                        
    Raises:
        IOError: If there's an error writing the file.
        pickle.PicklingError: If the model object cannot be pickled.
    """

    directory = "modelos"
    os.makedirs(directory, exist_ok=True)

    filepath = os.path.join(directory, filename)

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model successfully saved to {filepath}")
    except (IOError, pickle.PicklingError) as e:
        print(f"Error saving model to {filepath}: {e}")
        raise


def load_model(filename: str) -> Any:
    """
    Loads a model object from a file saved using pickle.

    Args:
        filename (str): The name of the file containing the saved model 
                        (assumed to be in the 'modelos' directory).

    Returns:
        Any: The loaded model object.

    Raises:
        IOError: If the file cannot be opened or read.
        pickle.UnpicklingError: If the file content is not a valid pickle stream 
                                or if there are issues deserializing the object 
                                (e.g., missing class definitions).
    """
    directory = "modelos"
    filepath = os.path.join(directory, filename)

    try:
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"Model successfully loaded from {filepath}")
        return loaded_model
    except (IOError, pickle.UnpicklingError) as e:
        print(f"Error loading model from {filepath}: {e}")
        raise