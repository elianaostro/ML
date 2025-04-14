import numpy as np
import pandas as pd

def Kmeans(df: pd.DataFrame, n_clusters: int = 3, max_iter: int = 100, random_state: Optional[int] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Performs K-Means clustering on a Pandas DataFrame.

    Initializes centroids by randomly choosing distinct samples from the DataFrame.
    Iteratively assigns samples to the nearest centroid and updates centroids 
    until convergence or max_iter is reached.

    Args:
        df (pd.DataFrame): Input DataFrame with numerical features.
        n_clusters (int, optional): The number of clusters to form. Defaults to 3.
        max_iter (int, optional): Maximum number of iterations for the algorithm. 
                                  Defaults to 100.
        random_state (Optional[int], optional): Seed for random number generation 
                                                for centroid initialization. Defaults to None.

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: A tuple containing:
            - labels (np.ndarray): Cluster labels assigned to each sample in the input df 
                                   (corresponding to the original order).
            - centroids_df (pd.DataFrame): DataFrame containing the final centroid positions,
                                           with columns matching the input df.
                                           
    Raises:
        ValueError: If n_clusters is less than 1 or greater than the number of samples.
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = df.values
    n_samples, n_features = X.shape

    initial_centroid_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[initial_centroid_indices]

    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iter):
        distances = np.sqrt(np.sum((X - centroids[:, np.newaxis, :])**2, axis=2))
        new_labels = np.argmin(distances, axis=0)

        if iteration > 0 and np.array_equal(new_labels, labels):
             print(f"KMeans converged early at iteration {iteration} (labels did not change).")
             break

        labels = new_labels

        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            cluster_samples = X[labels == k]
            if len(cluster_samples) == 0:
                print(f"Warning: Cluster {k} became empty during iteration {iteration}. "
                      f"Keeping previous centroid.")
                random_idx = np.random.choice(n_samples)
                new_centroids[k] = X[random_idx]
            else:
                new_centroids[k] = cluster_samples.mean(axis=0)

        centroids = new_centroids

        if np.allclose(centroids, new_centroids, atol=1e-6): 
            break
            
    centroids_df = pd.DataFrame(centroids, columns=df.columns)

    return labels, centroids_df

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
