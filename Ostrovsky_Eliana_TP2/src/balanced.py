import numpy as np
import pandas as pd

def undersample_random(X_df, y_array, random_state=None):
    """
    Undersampling aleatorio manteniendo X como DataFrame y y como array.
    
    Parámetros:
    - X_df (pd.DataFrame): Características
    - y_array (np.ndarray): Target
    - random_state (int): Semilla aleatoria
    
    Retorna:
    - X_resampled (pd.DataFrame), y_resampled (np.ndarray): Datos balanceados
    """

    temp_df = X_df.copy()
    temp_df['__target__'] = y_array
    
    _, counts = np.unique(y_array, return_counts=True)
    min_count = np.min(counts)

    sampled = temp_df.groupby('__target__', group_keys=False)\
                     .apply(lambda x: x.sample(min_count, random_state=random_state))

    X_res = sampled.drop('__target__', axis=1).sample(frac=1, random_state=random_state)
    y_res = sampled['__target__'].values
    
    return X_res, y_res

def oversample_duplicate(X_df, y_array, random_state=None):
    """
    Oversampling mediante duplicación manteniendo X como DataFrame y y como array.
    
    Parámetros:
    - X_df (pd.DataFrame): Características
    - y_array (np.ndarray): Target
    - random_state (int): Semilla aleatoria
    
    Retorna:
    - X_resampled (pd.DataFrame), y_resampled (np.ndarray): Datos balanceados
    """
    # Crear DataFrame temporal con el target
    temp_df = X_df.copy()
    temp_df['__target__'] = y_array
    
    # Calcular el tamaño máximo de clase
    max_count = pd.Series(y_array).value_counts().max()
    
    # Función de muestreo por grupo
    def sample_group(group):
        if len(group) < max_count:
            n_needed = max_count - len(group)
            additional = group.sample(n_needed, replace=True, random_state=random_state)
            return pd.concat([group, additional])
        return group
    
    # Aplicar muestreo
    sampled = temp_df.groupby('__target__', group_keys=False)\
                     .apply(sample_group)
    
    # Separar y reordenar
    X_res = sampled.drop('__target__', axis=1).sample(frac=1, random_state=random_state)
    y_res = sampled['__target__'].values
    
    return X_res, y_res
def simple_smote(X_df, y_array, k=5, random_state=None):
    """
    Implementación simplificada de SMOTE para DataFrames.
    
    Parámetros:
    - X_df (pd.DataFrame): Características (DataFrame)
    - y_array (np.ndarray): Etiquetas (array 1D)
    - k (int): Número de vecinos para SMOTE
    - random_state (int): Semilla aleatoria
    
    Retorna:
    - X_resampled (pd.DataFrame), y_resampled (np.ndarray): Datos balanceados
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convertir a numpy para cálculos
    X_np = X_df.values
    classes, counts = np.unique(y_array, return_counts=True)
    max_count = np.max(counts)
    
    X_resampled = [X_np]
    y_resampled = [y_array]
    
    for class_idx, count in zip(classes, counts):
        if count < max_count:
            # Obtener muestras de la clase actual
            mask = y_array == class_idx
            X_class = X_np[mask]
            n_needed = max_count - count
            
            # Calcular matriz de distancias
            distances = np.sqrt(((X_class[:, np.newaxis] - X_class)**2).sum(axis=2))
            np.fill_diagonal(distances, np.inf)
            
            # Obtener índices de los k vecinos más cercanos
            knn_indices = np.argpartition(distances, k, axis=1)[:, :k]
            
            synthetic_samples = []
            for _ in range(n_needed):
                # Seleccionar muestra base aleatoria
                sample_idx = np.random.randint(X_class.shape[0])
                neighbor_idx = np.random.choice(knn_indices[sample_idx])
                
                # Generar muestra sintética
                alpha = np.random.random()
                synthetic = X_class[sample_idx] + alpha * (X_class[neighbor_idx] - X_class[sample_idx])
                
                synthetic_samples.append(synthetic)
            
            # Convertir a DataFrame manteniendo nombres de columnas
            synthetic_df = pd.DataFrame(
                np.array(synthetic_samples),
                columns=X_df.columns
            )
            
            # Concatenar con datos originales
            X_resampled.append(synthetic_df.values)
            y_resampled.append(np.full(n_needed, class_idx))
    
    # Combinar y mezclar todos los datos
    X_combined = pd.DataFrame(
        np.concatenate(X_resampled),
        columns=X_df.columns
    )
    y_combined = np.concatenate(y_resampled)
    
    # Mezclar los datos
    indices = np.random.permutation(len(X_combined))
    return X_combined.iloc[indices].reset_index(drop=True), y_combined[indices]