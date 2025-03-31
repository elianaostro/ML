import numpy as np

class LinearRegression:
    """
    A simple linear regression model that supports training using the pseudoinverse method
    and gradient descent with optional L1 and L2 regularization.
    Attributes:
        X (np.ndarray): The input feature matrix with a bias term added.
        y (np.ndarray): The target values.
        coef (np.ndarray): The coefficients of the linear regression model.
        feature_names (list): The names of the features including the bias term.
    """

    def __init__(self, X, y):
        """
        Initializes the LinearRegression model with input features and target values.
        Args:
            X (pd.DataFrame or np.ndarray): The input feature matrix.
            y (np.ndarray): The target values.
        """
        self.X = np.c_[np.ones((X.shape[0], 1)), X] 
        self.y = y
        self.coef = None
        self.feature_names = ['bias'] + list(X.columns)

    def train_pseudoinverse(self):
        """
        Trains the linear regression model using the pseudoinverse method.
        """
        self.coef = np.linalg.pinv(self.X) @ self.y

    def train_gradient_descent(self, lr=0.0001, max_iters=10000, tolerance=1e-6, l1_lambda=0, l2_lambda=0):
        """
        Trains the linear regression model using gradient descent with optional L1 and L2 regularization.
        Args:
            lr (float): The learning rate for gradient descent. Default is 0.0001.
            max_iters (int): The maximum number of iterations for gradient descent. Default is 10000.
            tolerance (float): The tolerance for the stopping criterion. Default is 1e-6.
            l1_lambda (float): The regularization strength for L1 regularization. Default is 0.
            l2_lambda (float): The regularization strength for L2 regularization. Default is 0.
        """
        n_samples, n_features = self.X.shape
        self.coef = np.zeros(n_features)

        loss = float('inf')
        for i in range(max_iters):
            y_pred = self.X @ self.coef
            pred_diff = self.y - y_pred

            grad = -2 * (self.X.T @ pred_diff) / n_samples
            grad += l1_lambda * np.sign(self.coef)
            grad += l2_lambda * self.coef

            new_loss = np.mean(pred_diff ** 2)

            self.coef -= lr * grad

            if np.abs(loss - new_loss) < tolerance:
                break

            loss = new_loss

    def predict(self, X_new):
        """
        Predicts target values for new input features using the trained model.
        Args:
            X_new (pd.DataFrame or np.ndarray): The new input feature matrix.
        Returns:
            np.ndarray: The predicted target values.
        """
        X_new = np.c_[np.ones((X_new.shape[0], 1)), X_new]
        return X_new @ self.coef

    def print_coefficients(self):
        """
        Prints the coefficients of the trained linear regression model.
        """
        print("Model Coefficients:")
        for name, coef in zip(self.feature_names, self.coef.flatten()):
            print(f"{name}: {coef:.2f}")
        print()
