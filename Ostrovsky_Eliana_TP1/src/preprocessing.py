#funciones para el preprocesamiento
import numpy as np
import pandas as pd


def one_hot_encoder():
    pass

def handle_missing_values():
    pass

def convertir_unidades(df):
    df.loc[df['area_units'] == 'sqft', 'area'] *= 0.092903
    df['area_m2'] = df.apply(lambda row: row['area'] * 0.092903 if row['area_units'] == 'sqft' else row['area'], axis=1)
    return df

# Función para asignar el número de habitaciones basado en el área usando KNN

def knn_value(base_row, df, target_col, feature_cols, k):
    # Calcular la diferencia absoluta entre el valor base y los valores en el dataframe
    df['distance'] = np.linalg.norm(df[feature_cols].values - base_row[feature_cols].values, axis=1)
    
    # Seleccionar las k filas con la menor distancia
    nearest_neighbors = df.nsmallest(k, 'distance')
    
    # Obtener la moda de los valores target_col de los vecinos mas cercanos
    most_common_value = nearest_neighbors[target_col].dropna().mode()
    
    # Eliminar la columna temporal 'distance'
    df.drop(columns=['distance'], inplace=True)
    
    return most_common_value[0] if len(most_common_value) > 0 else np.nan

def assign_rooms_knn(area, df, k=5):
    base_row = pd.DataFrame({'area': [area]})
    return knn_value(base_row.iloc[0], df, 'rooms', ['area'], k)



def normalize(X, means=None, stds=None):
    """
    Normaliza las columnas especificadas de X usando la media y std proporcionadas.
    Si no se proporcionan, se calculan a partir de X.

    Parámetros:
    - X: DataFrame a normalizar.
    - means: Media para la normalización.
    - stds: Desviación estándar para la normalización.

    Retorna:
    - X normalizado.
    - Media y std usadas para la normalización.
    """
    cols = ["area", "age", "rooms","lat","lon"]
    
    if means is None or stds is None:
        means = X[cols].mean()
        stds = X[cols].std()
    
    X[cols] = (X[cols] - means) / stds
    return X, means, stds
