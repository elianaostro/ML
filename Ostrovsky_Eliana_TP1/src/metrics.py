#Funciones paa calcular metricas
import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def metrics(y_true, y_pred):
    results = {
        'mse': mse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'r2': r2(y_true, y_pred)
    }
    for metric, value in results.items():
        print(f"{metric}: {value}")
    return results