import pandas as pd

#Funciones para dividir los datos

def cross_val():
    pass

def split_data(df, target_column, train_ratio=0.8, random_state=None):
    """
    Divide el DataFrame en conjuntos de entrenamiento y validación.

    Parámetros:
    - df: DataFrame a dividir.
    - target_column: Nombre de la columna objetivo.
    - train_ratio: Proporción de datos para el conjunto de entrenamiento (default 0.8).
    - random_state: Semilla para el generador de números aleatorios (default None).

    Retorna:
    - X_train, X_val, y_train, y_val: Conjuntos de entrenamiento y validación.
    """
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)  # Mezclar el DataFrame
    X = df.drop(columns=[target_column])
    y = df[target_column].values
    train_size = int(train_ratio * len(X))

    X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    return X_train, X_val, y_train, y_val
