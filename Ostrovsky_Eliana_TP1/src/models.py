import numpy as np

class LinearRegression:
    def __init__(self, X, y):
        # Agregar término de sesgo
        bias = np.ones((X.shape[0], 1))
        self.X = np.c_[bias, X]
        self.y = y
        self.coef = None
    
    def train_pseudoinverse(self):
        """Entrena el modelo usando la pseudo-inversa"""
        self.coef = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
    
    def train_gradient_descent(self, lr=0.01, epochs=1000, clip_value=1e-2):
        """Entrena el modelo usando descenso por gradiente"""
        m, n = self.X.shape
        self.coef = np.zeros((n, 1))
        
        for _ in range(epochs):
            gradients = (2/m) * self.X.T @ (self.X @ self.coef - self.y)
            gradients = np.clip(gradients, -clip_value, clip_value)  # Clip gradients to avoid overflow
            self.coef -= lr * gradients
    
    def predict(self, X_new):
        """Realiza predicciones con el modelo entrenado"""
        X_new = np.c_[np.ones((X_new.shape[0], 1)), X_new]
        return X_new @ self.coef
    
    def print_coefficients(self):
        """Imprime los coeficientes del modelo"""
        print("Coeficientes del modelo:")
        for name, coef in zip(["intercept"] + [f"x{i}" for i in range(1, self.X.shape[1])], self.coef):
            print(f"{name}: {coef[0]}")
