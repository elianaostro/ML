import numpy as np

class LogisticRegression:
    """
    Logistic Regression model with L2 regularization.
    - Uses sigmoid for binary classification.
    - Uses softmax for multiclass classification.
    """
    def __init__(self, X, y):
        """
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
        """
        X = np.array(X)  # asegura que X es un array NumPy
        self.X = np.c_[np.ones((X.shape[0], 1)), X]  # agrega bias
        self.y = np.array(y)
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.feature_names = ['bias'] + (list(X.columns) if hasattr(X, 'columns') else [f'x{i}' for i in range(X.shape[1])])
        self.coef = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)  # estabilidad numérica
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def train(self, lr=0.01, max_iters=10000, tolerance=1e-6, l2_lambda=0.1):
        n_samples, n_features = self.X.shape

        if self.n_classes == 2:
            # Binaria: usamos 1 solo vector de coeficientes
            self.coef = np.zeros(n_features)
            y_binary = (self.y == self.classes[1]).astype(int)

            for _ in range(max_iters):
                z = self.X @ self.coef
                y_pred = self._sigmoid(z)
                error = y_pred - y_binary

                grad = (self.X.T @ error) / n_samples
                grad += l2_lambda * np.r_[0, self.coef[1:]]

                coef_new = self.coef - lr * grad
                if np.linalg.norm(self.coef - coef_new, ord=1) < tolerance:
                    break
                self.coef = coef_new

        else:
            # Multiclase: One-vs-Rest con softmax (todos juntos)
            self.coef = np.zeros((self.n_classes, n_features))
            Y_onehot = np.eye(self.n_classes)[np.searchsorted(self.classes, self.y)]

            for _ in range(max_iters):
                logits = self.X @ self.coef.T  # shape: (n_samples, n_classes)
                probs = self._softmax(logits)
                error = probs - Y_onehot

                grad = (error.T @ self.X) / n_samples
                grad += l2_lambda * np.c_[np.zeros((self.n_classes, 1)), self.coef[:, 1:]]  # no regularizamos el bias

                coef_new = self.coef - lr * grad
                if np.linalg.norm(self.coef - coef_new, ord=1) < tolerance:
                    break
                self.coef = coef_new

    def predict_proba(self, X_new):
        X_new = np.c_[np.ones((X_new.shape[0], 1)), X_new]

        if self.n_classes == 2:
            probs = self._sigmoid(X_new @ self.coef)
            return probs.reshape(-1, 1)  # forma (n_samples, 1)
        else:
            logits = X_new @ self.coef.T
            return self._softmax(logits)

    def predict(self, X_new):
        probs = self.predict_proba(X_new)
        if self.n_classes == 2:
            return (probs >= 0.5).astype(int).flatten()
        else:
            return self.classes[np.argmax(probs, axis=1)]

    def print_coefficients(self):
        if self.n_classes == 2:
            print("Binary Logistic Regression Coefficients:")
            for name, coef in zip(self.feature_names, self.coef):
                print(f"{name}: {coef:.4f}")
        else:
            print("Multiclass Logistic Regression Coefficients:")
            for idx, c in enumerate(self.classes):
                print(f"\nClass {c}:")
                for name, coef in zip(self.feature_names, self.coef[idx]):
                    print(f"{name}: {coef:.4f}")
