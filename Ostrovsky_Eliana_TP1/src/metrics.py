import numpy as np

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def calculate_rmse(y_true, y_pred):
    mse = calculate_mse(y_true, y_pred)
    return np.sqrt(mse)

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def print_metrics(y_true, y_pred, label):
    """
    Print various regression metrics for the given true and predicted values.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    label (str): Label to identify the metrics output.

    Returns:
    None
    """
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    r_2 = calculate_r2(y_true, y_pred)
    print(f"{label} Metrics:")
    print(f"- MSE: {mse:.2f}")
    print(f"- RMSE: {rmse:.2f}")
    print(f"- MAE: {mae:.2f}")
    print(f"- R^2: {r_2:.2f}\n")