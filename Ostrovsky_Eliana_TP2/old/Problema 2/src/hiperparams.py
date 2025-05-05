from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.metrics import print_metrics
from src.models import LinearRegression
from src.preprocessing import normalize, split_and_normalize


def plot_regularization_coefficients(df, lambdas, reg_type="ridge"):
    """
    Plots the regression coefficients as a function of lambda for Ridge (L2) or LASSO (L1).

    Parameters:
    - df: DataFrame with the data
    - lambdas: List of lambda values
    - reg_type: "ridge" for L2 or "lasso" for L1
    """
    df, _, _ = normalize(df)
    X = df.drop(columns=['price'])
    y = df['price'].to_numpy()
    coefficients = []
    feature_names = ['bias'] + list(X.columns)

    for lmbd in lambdas:
        model = LinearRegression(X, y)

        if reg_type == "ridge":
            model.train_gradient_descent(lr=0.0001, l2_lambda=lmbd)
        elif reg_type == "lasso":
            model.train_gradient_descent(lr=0.0001, l1_lambda=lmbd)

        coefficients.append(model.coef.flatten())

    coefficients = np.array(coefficients)

    plt.figure(figsize=(10, 6))
    for i in range(coefficients.shape[1]):
        plt.plot(lambdas, coefficients[:, i], label=feature_names[i])

    plt.xscale("log")
    plt.xlabel("Lambda (Regularization Strength)")
    plt.ylabel("Coefficient Value")
    plt.title(f"{'Ridge (L2)' if reg_type == 'ridge' else 'LASSO (L1)'} Regression Coefficients vs Lambda")
    plt.legend()
    plt.show()

def find_best_lambda_validation(df, lambdas):
    """
    Finds the best lambda value using simple validation by minimizing the MSE.

    Parameters:
    - df: DataFrame with the data
    - lambdas: List of lambda values to evaluate

    Returns:
    - best_lambda: Best lambda value found
    - mse_values: List of MSE values obtained for each lambda
    """
    X_train, X_val, y_train, y_val = split_and_normalize(df)
    mse_values = []
    
    for lmbd in lambdas:
        model = LinearRegression(X_train, y_train)
        model.train_gradient_descent(lr=0.00001, l2_lambda=lmbd)
        predictions = model.predict(X_val)
        mse = np.mean((y_val - predictions) ** 2)
        mse_values.append(mse)
    
    best_lambda = lambdas[np.argmin(mse_values)]
    
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, mse_values, marker='o', linestyle='-')
    plt.xscale("log")
    plt.xlabel("Lambda (Regularization Strength)")
    plt.ylabel("MSE (Validation)")
    plt.title("MSE vs Lambda (Validation Set)")
    plt.show()
    
    return best_lambda, mse_values


def find_best_lambda_cross_validation(df, lambdas, k=5):
    """
    Finds the best lambda value using cross-validation by minimizing the MSE.

    Parameters:
    - df: DataFrame with the data
    - lambdas: List of lambda values to evaluate
    - k: Number of folds in cross-validation

    Returns:
    - best_lambda: Best lambda value found
    - mean_mse_values: List of mean MSE values for each lambda
    """
    X, _, _ = normalize(df.drop(columns=['price']))
    y = df['price']
    n = len(y)
    fold_size = n // k
    mean_mse_values = []

    for lmbd in lambdas:
        mse_fold_values = []

        for i in range(k):
            start, end = i * fold_size, (i + 1) * fold_size
            X_val_fold, y_val_fold = X[start:end], y[start:end]
            X_train_fold = np.concatenate([X[:start], X[end:]], axis=0)
            y_train_fold = np.concatenate([y[:start], y[end:]], axis=0)

            model = LinearRegression(pd.DataFrame(X_train_fold, columns=X.columns), y_train_fold)
            model.train_gradient_descent(lr=0.001, l2_lambda=lmbd)
            predictions = model.predict(X_val_fold).flatten()
            mse = np.mean((y_val_fold - predictions) ** 2)
            mse_fold_values.append(mse)

        mean_mse = np.mean(mse_fold_values)
        mean_mse_values.append(mean_mse)

    best_lambda = lambdas[np.argmin(mean_mse_values)]

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, mean_mse_values, marker='o', linestyle='-')
    plt.xscale("log")
    plt.xlabel("Lambda (Regularization Strength)")
    plt.ylabel("Mean MSE (Cross-Validation)")
    plt.title("Mean MSE vs Lambda (Cross-Validation)")
    plt.show()

    return best_lambda, mean_mse_values
