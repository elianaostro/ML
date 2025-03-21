import numpy as np

class LinearRegression:
    def __init__(self, X, y):
        self.X = np.c_[np.ones((X.shape[0], 1)), X]  # Agregar término de sesgo
        self.y = y.reshape(-1, 1)
        self.coef = None
    
    def train_pseudoinverse(self):
        """Entrena el modelo usando la pseudo-inversa"""
        self.coef = np.linalg.pinv(self.X) @ self.y
    
    def train_gradient_descent(self, lr=0.01, epochs=1000):
        """Entrena el modelo usando descenso por gradiente"""
        m, n = self.X.shape
        self.coef = np.zeros((n, 1))
        
        for _ in range(epochs):
            gradients = (2/m) * self.X.T @ (self.X @ self.coef - self.y)
            self.coef -= lr * gradients
    
    def predict(self, X_new):
        """Realiza predicciones con el modelo entrenado"""
        X_new = np.c_[np.ones((X_new.shape[0], 1)), X_new]
        return X_new @ self.coef
    
    def print_coefficients(self, feature_names):
        """Imprime los coeficientes de la regresión con los nombres de las variables"""
        print("Intercepto:", self.coef[0][0])
        for name, coef in zip(feature_names, self.coef[1:]):
            print(f"{name}: {coef[0]}")

# Función de error cuadrático medio (ECM)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
