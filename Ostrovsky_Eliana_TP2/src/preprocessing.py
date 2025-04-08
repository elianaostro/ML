import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Divide los datos en conjuntos de entrenamiento y prueba"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    test_samples = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_idx = indices[:test_samples]
    train_idx = indices[test_samples:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standard_scaler(X):
    """Normalización estándar (z-score)"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0  # Evitar división por cero
    return (X - mean) / std, mean, std

def minmax_scaler(X):
    """Normalización min-max"""
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0  # Evitar división por cero
    return (X - min_vals) / range_vals, min_vals, max_vals

def handle_missing_values(X, strategy='mean'):
    """Manejo de valores faltantes"""
    if strategy == 'mean':
        fill_value = np.nanmean(X, axis=0)
    elif strategy == 'median':
        fill_value = np.nanmedian(X, axis=0)
    elif strategy == 'zero':
        fill_value = 0
    else:
        raise ValueError("Estrategia no válida")
    
    return np.where(np.isnan(X), fill_value, X)

def oversample_duplicate(X, y, random_state=None):
    """Oversampling mediante duplicación"""
    if random_state is not None:
        np.random.seed(random_state)
    
    classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    
    X_resampled = [X]
    y_resampled = [y]
    
    for class_idx, count in zip(classes, counts):
        if count < max_count:
            X_class = X[y == class_idx]
            y_class = y[y == class_idx]
            
            n_needed = max_count - count
            indices = np.random.choice(X_class.shape[0], size=n_needed, replace=True)
            
            X_resampled.append(X_class[indices])
            y_resampled.append(y_class[indices])
    
    return np.concatenate(X_resampled), np.concatenate(y_resampled)

def undersample_random(X, y, random_state=None):
    """Undersampling aleatorio"""
    if random_state is not None:
        np.random.seed(random_state)
    
    classes, counts = np.unique(y, return_counts=True)
    min_count = np.min(counts)
    
    X_resampled = []
    y_resampled = []
    
    for class_idx in classes:
        X_class = X[y == class_idx]
        y_class = y[y == class_idx]
        
        indices = np.random.choice(X_class.shape[0], size=min_count, replace=False)
        
        X_resampled.append(X_class[indices])
        y_resampled.append(y_class[indices])
    
    return np.concatenate(X_resampled), np.concatenate(y_resampled)

def simple_smote(X, y, k=5, random_state=None):
    """Implementación simplificada de SMOTE"""
    if random_state is not None:
        np.random.seed(random_state)
    
    classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    
    X_resampled = [X]
    y_resampled = [y]
    
    for class_idx, count in zip(classes, counts):
        if count < max_count:
            X_class = X[y == class_idx]
            n_needed = max_count - count
            
            # Encontrar k vecinos más cercanos para cada muestra
            distances = np.sqrt(((X_class[:, np.newaxis] - X_class)**2).sum(axis=2))
            np.fill_diagonal(distances, np.inf)
            knn_indices = np.argpartition(distances, k, axis=1)[:, :k]
            
            synthetic_samples = []
            for i in range(n_needed):
                # Seleccionar una muestra aleatoria
                sample_idx = np.random.randint(X_class.shape[0])
                # Seleccionar un vecino aleatorio
                neighbor_idx = np.random.choice(knn_indices[sample_idx])
                
                # Crear muestra sintética
                diff = X_class[neighbor_idx] - X_class[sample_idx]
                alpha = np.random.random()
                synthetic = X_class[sample_idx] + alpha * diff
                
                synthetic_samples.append(synthetic)
            
            X_resampled.append(np.array(synthetic_samples))
            y_resampled.append(np.full(n_needed, class_idx))
    
    return np.concatenate(X_resampled), np.concatenate(y_resampled)