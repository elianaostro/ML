import numpy as np
import pandas as pd

def clean_data(df):
    df['CellAdhesion'] = df['CellAdhesion'].apply(lambda x: x if 0 <= x <= 1 else np.nan)
    df['NuclearMembrane'] = df['NuclearMembrane'].apply(lambda x: x if 1 <= x <= 5 else np.nan)
    df['OxygenSaturation'] = df['OxygenSaturation'].apply(lambda x: x if 0 <= x <= 100 else np.nan)
    df['Vascularization'] = df['Vascularization'].apply(lambda x: x if 0 <= x <= 10 else np.nan)
    df['InflammationMarkers'] = df['InflammationMarkers'].apply(lambda x: x if 0 <= x <= 100 else np.nan)

    df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).apply(
        lambda col: col.map(lambda x: np.nan if x < 0 else x)
    )

    numeric_columns = [
        "CellSize", "CellShape", "NucleusDensity", "ChromatinTexture",
        "CytoplasmSize", "CellAdhesion", "MitosisRate", "NuclearMembrane",
        "GrowthFactor", "OxygenSaturation", "Vascularization", "InflammationMarkers"
    ]
    Q1 = df[numeric_columns].quantile(0.05)
    Q3 = df[numeric_columns].quantile(0.95)
    IQR = Q3 - Q1

    for column in numeric_columns:
        df[column] = df[column].mask(
            (df[column] < (Q1[column] - 1.5 * IQR[column])) | 
            (df[column] > (Q3[column] + 1.5 * IQR[column]))
        )
        
    df['NaN_Count'] = df.isna().sum(axis=1)
    df = df[df['NaN_Count'] < 7]
    return df.drop(columns=['NaN_Count'])

def handle_categorical_features(df):
    df['Epthlial'] = np.where(df['CellType'].isna(), np.nan, (df['CellType'] == 'Epthlial').astype(int))
    df['Mesnchymal'] = np.where(df['CellType'].isna(), np.nan, (df['CellType'] == 'Mesnchymal').astype(int))
    df['GeneticMutation'] = (df['GeneticMutation'] == 'Presnt').astype(int)
    df = df.drop(columns=["CellType"])
    return df

def detect_outliers_iqr(X, factor=1.5):
    """
    Detecta outliers usando el método del rango intercuartílico (IQR).
    
    Parámetros:
    - X (np.array): Datos de características
    - factor (float): Factor multiplicador del IQR (típicamente 1.5)
    
    Retorna:
    - Máscara booleana de outliers (True = outlier)
    """
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return (X < lower_bound) | (X > upper_bound)

def knn_value(base_row, df, target_col, feature_cols, k):
    """
    Calcula el valor usando k-vecinos más cercanos para una fila dada.

    Parámetros:
    base_row (pd.Series): La fila para la cual calcular el valor de k-vecinos.
    df (pd.DataFrame): El DataFrame que contiene los datos.
    target_col (str): Nombre de la columna objetivo cuyo valor se quiere predecir.
    feature_cols (list): Lista de nombres de columnas para calcular la distancia.
    k (int): Número de vecinos más cercanos a considerar.

    Returns:
    El valor más común entre los k-vecinos más cercanos en la columna objetivo.
    Si no hay un valor común, devuelve NaN.
    """

    df_copy = df.copy()
    df_copy['distance'] = np.linalg.norm(df_copy[feature_cols].values - base_row[feature_cols].values, axis=1)
    nearest_neighbors = df_copy.nsmallest(k, 'distance')
    most_common_value = nearest_neighbors[target_col].dropna().mode()
    df_copy.drop(columns=['distance'], inplace=True)
    
    return most_common_value[0] if not most_common_value.empty else np.nan

def normalize(X, means=None, stds=None, exclude_cols=None):
    """
    Normaliza columnas numéricas en el dataset.
    
    Args:
        X (pd.DataFrame): DataFrame a normalizar
        means (pd.Series, optional): Medias a usar para normalización
        stds (pd.Series, optional): Desviaciones estándar a usar para normalización
        exclude_cols (list, optional): Columnas a excluir de la normalización
        
    Returns:
        X_norm (pd.DataFrame): DataFrame normalizado
        means (pd.Series): Medias usadas para normalización
        stds (pd.Series): Desviaciones estándar usadas para normalización
    """
    X_norm = X.copy()
    
    # Determinar qué columnas normalizar
    if exclude_cols is None:
        exclude_cols = []
    
    # Seleccionar columnas numéricas excluyendo las especificadas
    numeric_cols = X.select_dtypes(include=np.number).columns.difference(exclude_cols)
    
    # Si no se proporcionan medias y desviaciones, calcularlas
    if means is None:
        means = X[numeric_cols].mean()
    if stds is None:
        stds = X[numeric_cols].std()
    
    # Normalizar
    X_norm[numeric_cols] = (X[numeric_cols] - means) / (stds + 1e-8)
    
    return X_norm, means, stds

def handle_missing_values(df, strategy='mean', train_df=None, k=5):
    """
    Maneja valores faltantes en un DataFrame usando múltiples estrategias.
    
    Args:
        df (pd.DataFrame): DataFrame con posibles valores faltantes
        strategy (str): Estrategia de imputación ('mean', 'median', 'zero', 'knn')
        train_df (pd.DataFrame, optional): DataFrame de entrenamiento para imputar en validación
        k (int): Número de vecinos para KNN (solo si strategy='knn')
        
    Returns:
        pd.DataFrame: DataFrame con valores faltantes imputados
    """
    df_imputed = df.copy()
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    valid_numeric_cols = [col for col in numeric_cols if not df[col].isnull().all()]
    
    reference_df = train_df if train_df is not None else df
    
    if strategy == 'mean':
        means = reference_df[valid_numeric_cols].mean()
        for col in valid_numeric_cols:
            df_imputed[col].fillna(means[col], inplace=True)

    elif strategy == 'knn':
        combined_df = pd.concat([reference_df, df_imputed]) if train_df is not None else df_imputed
        
        for col in valid_numeric_cols:
            rows_with_nan = df_imputed[df_imputed[col].isnull()].index
            feature_cols = [c for c in valid_numeric_cols if c != col]
            
            if not feature_cols:
                mean_value = reference_df[col].mean()
                df_imputed.loc[rows_with_nan, col] = mean_value
                continue
            
            for idx in rows_with_nan:
                current_row = df_imputed.loc[idx]
                known_df = combined_df[~combined_df[col].isnull()]
                if known_df.empty:
                    mean_value = reference_df[col].mean()
                    df_imputed.loc[idx, col] = mean_value
                else:
                    knn_result = knn_value(current_row, known_df, col, feature_cols, k)
                    df_imputed.loc[idx, col] = knn_result
    return df_imputed

def split_data(df, target_column, train_ratio=0.8, random_state=None):
    '''
    Split the DataFrame into training and validation sets.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    train_ratio (float): The ratio of training data to total data.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    X_train (pandas.DataFrame): The training features.
    X_val (pandas.DataFrame): The validation features.
    y_train (np.ndarray): The training target values.
    y_val (np.ndarray): The validation target values.
    '''
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    train_size = int(train_ratio * len(df_shuffled))
    
    X_train = df_shuffled.iloc[:train_size].drop(columns=[target_column])
    X_val = df_shuffled.iloc[train_size:].drop(columns=[target_column])
    
    y_train = df_shuffled.iloc[:train_size][target_column].values
    y_val = df_shuffled.iloc[train_size:][target_column].values
    
    return X_train, X_val, y_train, y_val


def split_and_normalize(df, target_column, exclude_cols=None, train_ratio=0.8, random_state=None):
    """
    Divide los datos y normaliza las características.
    
    Args:
        df (pd.DataFrame): DataFrame a dividir y normalizar
        target_column (str): Nombre de la columna objetivo
        exclude_cols (list, optional): Columnas a excluir de la normalización
        train_ratio (float): Proporción de datos para entrenamiento
        random_state (int): Semilla aleatoria para reproducibilidad
        
    Returns:
        X_train (pd.DataFrame): Características de entrenamiento normalizadas
        X_val (pd.DataFrame): Características de validación normalizadas
        y_train (np.ndarray): Valores objetivo de entrenamiento
        y_val (np.ndarray): Valores objetivo de validación
        means (pd.Series): Medias usadas para normalización
        stds (pd.Series): Desviaciones estándar usadas para normalización
    """
    # Dividir los datos
    X_train, X_val, y_train, y_val = split_data(df, target_column, train_ratio, random_state)
    
    # Normalizar
    X_train_norm, means, stds = normalize(X_train, exclude_cols=exclude_cols)
    X_val_norm, _, _ = normalize(X_val, means=means, stds=stds, exclude_cols=exclude_cols)

    X_train_norm = handle_missing_values(X_train_norm, strategy='knn', train_df=X_train)
    X_val_norm = handle_missing_values(X_val_norm, strategy='knn', train_df=X_train)
    
    return X_train_norm, X_val_norm, y_train, y_val, means, stds

def stratified_split(X, y, train_ratio=0.8, random_state=None):
    """
    División estratificada de datos en conjuntos de entrenamiento y prueba.
    Mantiene la proporción de clases en ambos conjuntos.
    
    Parámetros:
    - X: Datos de características (numpy array o DataFrame)
    - y: Etiquetas (numpy array)
    - train_ratio: Proporción del conjunto de entrenamiento (0-1)
    - random_state: Semilla para reproducibilidad
    
    Retorna:
    - X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in classes}
    
    train_indices = []
    test_indices = []
    
    for cls in classes:
        indices = class_indices[cls]
        np.random.shuffle(indices)
        
        n_train = int(len(indices) * train_ratio)
        
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])
    
    # Mezclar los índices para evitar orden por clase
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices], \
           X.iloc[test_indices] if hasattr(X, 'iloc') else X[test_indices], \
           y[train_indices], y[test_indices]

def stratified_split_and_normalize(df, target_column, exclude_cols=None, train_ratio=0.8, random_state=None):
    """
    Divide los datos y normaliza las características.
    Args:
    df (pd.DataFrame): DataFrame a dividir y normalizar
    target_column (str): Nombre de la columna objetivo
    exclude_cols (list, optional): Columnas a excluir de la normalización
    train_ratio (float): Proporción de datos para entrenamiento
    random_state (int): Semilla aleatoria para reproducibilidad
    Returns:
    X_train (pd.DataFrame): Características de entrenamiento normalizadas
    X_val (pd.DataFrame): Características de validación normalizadas
    y_train (np.ndarray): Valores objetivo de entrenamiento
    y_val (np.ndarray): Valores objetivo de validación
    means (pd.Series): Medias usadas para normalización
    stds (pd.Series): Desviaciones estándar usadas para normalización
    """
    # Separar características y objetivo
    X = df.drop(columns=[target_column])
    y = df[target_column].values
    
    # Dividir los datos usando la función stratified_split
    X_train, X_val, y_train, y_val = stratified_split(X, y, 1-train_ratio, random_state)
    
    # Normalizar
    X_train_norm, means, stds = normalize(X_train, exclude_cols=exclude_cols)
    X_val_norm, _, _ = normalize(X_val, means=means, stds=stds, exclude_cols=exclude_cols)
    
    X_train_norm = handle_missing_values(X_train_norm, strategy='knn', train_df=X_train)
    X_val_norm = handle_missing_values(X_val_norm, strategy='knn', train_df=X_train)
    
    return X_train_norm, X_val_norm, y_train, y_val, means, stds