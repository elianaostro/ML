import numpy as np
from typing import Optional, Tuple


def stratified_split(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8, 
                     test_ratio: float = 0.2, random_state: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a stratified split of NumPy arrays into training and test sets.
    Maintains the same proportion of classes in both sets as in the original dataset.
    
    Args:
        X (np.ndarray): Array with features/data
        y (np.ndarray): Array with labels/classes
        train_ratio (float): Proportion of the dataset for the training set. Default is 0.8.
        test_ratio (float): Proportion of the dataset for the test set. Default is 0.2.
        random_state (Optional[int]): Seed for reproducibility. Default is None.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A tuple containing: X_train, X_test, y_train, y_test.
    
    Raises:
        ValueError: If the sum of ratios is not equal to 1.
    """
    if abs(train_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("The sum of train_ratio and test_ratio must equal 1")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    
    class_indices = {i: np.where(y_indices == i)[0] for i in range(n_classes)}
    
    train_indices = []
    test_indices = []
    
    for i in range(n_classes):
        indices_for_class = class_indices[i]
        n_class_samples = len(indices_for_class)
        
        n_train_class = int(np.round(n_class_samples * train_ratio))
        
        if n_class_samples >= 2:
            if n_train_class == 0:
                n_train_class = 1
        
        np.random.shuffle(indices_for_class)
        
        train_indices.extend(indices_for_class[:n_train_class])
        test_indices.extend(indices_for_class[n_train_class:])
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def custom_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """
    Función para calcular la matriz de confusión sin sklearn
    """
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        conf_matrix[y_true[i], y_pred[i]] += 1
    return conf_matrix
