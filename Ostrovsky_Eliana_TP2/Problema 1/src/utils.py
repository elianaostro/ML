import pandas as pd

from src.metrics import print_metrics
from src.models import LinearRegression
from src.preprocessing import split_and_normalize

def try_model(df, cols=['area_m2', 'is_house', 'has_pool', 'age', 'lat', 'lon', 'area_units'], method='pseudoinverse'):
    '''
    Trains a linear regression model on the given DataFrame and prints metrics for training and validation sets.
    
    Parameters:
    - df: DataFrame with the data
    - cols: List of columns to use as features
    - method: Training method to use ('pseudoinverse' or 'gradient')
    '''
    X_train, X_val, y_train, y_val = split_and_normalize(df)
    model = LinearRegression(X_train[cols], y_train)

    if method == 'pseudoinverse':
        print("Training with Pseudoinverse:")
        model.train_pseudoinverse()
    elif method == 'gradient':
        print("Training with Gradient Descent:")
        model.train_gradient_descent()
    else:
        raise ValueError("Invalid training method. Choose 'pseudoinverse' or 'gradient'.")

    print_metrics(y_train, model.predict(X_train[cols]), "Training")
    print_metrics(y_val, model.predict(X_val[cols]), "Validation")
    model.print_coefficients()