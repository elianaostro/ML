import pickle
from typing import Any

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
        raise


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
        raise