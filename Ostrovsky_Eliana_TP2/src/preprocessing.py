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

def knn_impute(X, k=5, metric='euclidean'):
    """
    Implementación básica de KNN Imputer
    
    Args:
        X: Matriz de datos con NaN (n_samples, n_features)
        k: Número de vecinos a considerar
        metric: Métrica de distancia ('euclidean' o 'manhattan')
        
    Returns:
        Matriz con valores imputados
    """
    X_imputed = X.copy()
    _, n_features = X.shape
    
    # Para cada feature con NaN
    for col in range(n_features):
        nan_mask = np.isnan(X[:, col])
        
        if not np.any(nan_mask):
            continue  # No hay NaN en esta columna
            
        # Indices de muestras con valores conocidos
        known_idx = np.where(~nan_mask)[0]
        unknown_idx = np.where(nan_mask)[0]
        
        # Para cada muestra con NaN
        for idx in unknown_idx:
            # Calcular distancias
            if metric == 'euclidean':
                distances = np.sqrt(np.sum((X[known_idx] - X[idx])**2, axis=1))
            elif metric == 'manhattan':
                distances = np.sum(np.abs(X[known_idx] - X[idx]), axis=1)
            else:
                raise ValueError("Métrica no soportada")
            
            # Encontrar k vecinos más cercanos
            nearest_indices = known_idx[np.argsort(distances)[:k]]
            
            # Imputar con la media de los vecinos (ignorando NaN en vecinos)
            valid_values = X[nearest_indices, col]
            valid_values = valid_values[~np.isnan(valid_values)]
            
            if len(valid_values) > 0:
                X_imputed[idx, col] = np.mean(valid_values)
            else:
                # Si todos los vecinos tienen NaN, usar la media global
                X_imputed[idx, col] = np.nanmean(X[:, col])
    
    return X_imputed

def knn_impute_fast(X, k=5):
    """Versión optimizada de KNN Imputer usando operaciones vectorizadas"""
    X_imputed = X.copy()
    _, n_features = X.shape
    
    for col in range(n_features):
        nan_mask = np.isnan(X[:, col])
        if not nan_mask.any():
            continue
            
        # Calcular distancias entre muestras con NaN y sin NaN
        known_samples = X[~nan_mask]
        unknown_samples = X[nan_mask]
        
        # Distancia euclidiana vectorizada
        distances = np.sqrt(((known_samples[:, np.newaxis] - unknown_samples)**2).sum(axis=2))
        
        # Para cada muestra con NaN, encontrar k vecinos más cercanos
        for i, sample_idx in enumerate(np.where(nan_mask)[0]):
            nearest_indices = np.argsort(distances[:, i])[:k]
            valid_values = known_samples[nearest_indices, col]
            valid_values = valid_values[~np.isnan(valid_values)]
            
            X_imputed[sample_idx, col] = np.mean(valid_values) if len(valid_values) > 0 else np.nanmean(X[:, col])
    
    return X_imputed

def handle_missing_values(X, strategy='mean', k=5, metric='euclidean'):
    """
    Manejo de valores faltantes con múltiples estrategias
    
    Args:
        X: Matriz de datos (n_samples, n_features)
        strategy: 'mean', 'median', 'zero', 'knn'
        k: Número de vecinos para KNN (solo si strategy='knn')
        metric: 'euclidean' o 'manhattan' (solo si strategy='knn')
        
    Returns:
        Matriz sin valores faltantes
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    if not np.isnan(X).any():
        return X  # No hay valores faltantes
    
    if strategy == 'mean':
        fill_value = np.nanmean(X, axis=0)
        return np.where(np.isnan(X), fill_value, X)
    elif strategy == 'median':
        fill_value = np.nanmedian(X, axis=0)
        return np.where(np.isnan(X), fill_value, X)
    elif strategy == 'zero':
        return np.where(np.isnan(X), 0, X)
    elif strategy == 'knn':
        return knn_impute(X, k=k, metric=metric)
    elif strategy == 'knn_fast':
        return knn_impute_fast(X, k=k)
    else:
        raise ValueError(f"Estrategia no válida: {strategy}. Use 'mean', 'median', 'zero' o 'knn'")


def detect_outliers_iqr(X, factor=1.5):
    """
    Detecta outliers usando el método del rango intercuartílico (IQR).
    
    Parámetros:
    - X: Datos de características (numpy array)
    - factor: Factor multiplicador del IQR (típicamente 1.5)
    
    Retorna:
    - Máscara booleana de outliers (True = outlier)
    """
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return (X < lower_bound) | (X > upper_bound)


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