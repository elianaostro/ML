import numpy as np

class LocallyWeightedRegression:
    """
    Locally Weighted Regression model.
    """
    def __init__(self, tau):
        """
        Constructor for Locally Weighted Regression with bandwidth parameter.

        Parameters:
        tau (float): Bandwidth of the kernel function.
        """
        self.tau = tau

    def get_K(self, query_x, X):
        """
        Calculate the diagonal weight matrix.

        Parameters:
        query_x (numpy.ndarray): Target x where we want to calculate 
                                 the locally weighted linear regression.
        X (numpy.ndarray): Training examples.

        Returns:
        numpy.matrix: Diagonal weight matrix.
        """
        N = X.shape[0]  # Number of examples
        K = np.asmatrix(np.eye(N))  # Initialize with an identity matrix
        for i in range(N):  # Calculate weights for the query point
            xi = X[i]
            K[i, i] = np.exp(np.dot((xi - query_x).T, xi - query_x) / (-2 * self.tau ** 2))
        return K

    def predict(self, X, Y, query_x):
        """
        Predict the output for a given input query_x.

        Parameters:
        X (numpy.ndarray): Training examples x values.
        Y (numpy.ndarray): Training examples y values.
        query_x (numpy.ndarray): Input query point.

        Returns:
        tuple: Tuple containing the parameters and the predicted value.
        """
        N = X.shape[0]
        X_ = np.hstack((X, np.ones((N, 1)))) # To incorporate bias term 
        qx = np.array([query_x, np.ones_like(query_x)]).T # To incorporate bias term 
        K = self.get_K(qx, X_)
        w = np.linalg.pinv(X_.T * (K * X_)) * (X_.T * (K * Y)) 
        pred = np.dot(qx, w)
        return w, pred