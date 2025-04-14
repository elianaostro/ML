import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any

def _mode_1d(a: np.ndarray) -> Union[int, float, Any]:
    """Finds the most frequent value in a 1D NumPy array."""
    vals, counts = np.unique(a, return_counts=True)
    return vals[np.argmax(counts)]

def KMeans(df: pd.DataFrame, n_clusters: int = 3, max_iter: int = 100, random_state: Optional[int] = None) -> Tuple[np.ndarray, pd.DataFrame]:
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
    Implements Logistic Regression supporting both binary and multi-class 
    classification (using Softmax Regression for multi-class) with L2 regularization.

    The model automatically detects the classification type based on the number of 
    unique classes in the target variable `y` during `fit`.

    Attributes:
        learning_rate (float): Step size for gradient descent updates.
        n_iterations (int): Number of gradient descent iterations.
        reg_lambda (float): L2 regularization strength (lambda).
        class_weight (Optional[Union[Dict[Any, float], str]]): Weights associated 
            with classes, primarily for binary classification. If 'balanced', 
            weights are adjusted inversely proportional to class frequencies 
            (only for binary). If a dict, maps class labels to weights (only for binary). 
            Not used in multi-class mode. Defaults to None.
        weights (Optional[np.ndarray]): Learned feature weights after fitting. 
            Shape [n_features] for binary, [n_features, n_classes] for multi-class.
        bias (Optional[Union[float, np.ndarray]]): Learned bias term(s) after fitting.
            Scalar float for binary, shape [n_classes] for multi-class.
        classes_ (Optional[np.ndarray]): Unique class labels encountered during fit 
            (shape [n_classes]). Sorted.
        mode_ (Optional[str]): Indicates the mode of operation: 'binary' or 'multinomial'. Set during fit.
    """
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, reg_lambda: float = 0.1, class_weight: Optional[Union[Dict[Any, float], str]] = None) -> None:
        """
        Initializes the unified LogisticRegression classifier.

        Args:
            learning_rate (float, optional): Step size for gradient descent. Defaults to 0.01.
            n_iterations (int, optional): Number of training iterations. Defaults to 1000.
            reg_lambda (float, optional): L2 regularization strength. Defaults to 0.1.
            class_weight (Optional[Union[Dict[Any, float], str]], optional): Class weights, 
                intended mainly for binary classification. Use 'balanced' or a dict 
                mapping labels to weights. Defaults to None. Will be ignored with a warning 
                in multi-class scenarios.
        """
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations
        self.reg_lambda: float = reg_lambda
        self.class_weight_config: Optional[Union[Dict[Any, float], str]] = class_weight 
        
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[Union[float, np.ndarray]] = None
        self.classes_: Optional[np.ndarray] = None
        self.mode_: Optional[str] = None 
        
        self._n_classes: Optional[int] = None
        self._actual_class_weights: Optional[Dict[Any, float]] = None
        self._class_map: Optional[Dict[Any, int]] = None 

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Applies the sigmoid function element-wise."""
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Applies the softmax function numerically stably along axis 1."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # def _compute_loss(self, h: np.ndarray, y: Union[np.ndarray, np.ndarray], weights: np.ndarray, n_samples: int) -> float:
    #     """Calculates the regularized loss based on the detected mode."""
    #     epsilon = 1e-15 
        
    #     l2_reg = (self.reg_lambda / (2 * n_samples)) * np.sum(weights**2)

    #     if self.mode_ == 'binary':
    #         loss = y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)
    #         if self._actual_class_weights is not None:
    #             weight_vector = np.array([self._actual_class_weights.get(int(label), 1.0) for label in y])
    #             cost = (-1 / n_samples) * np.sum(weight_vector * loss) + l2_reg
    #         else:
    #             cost = (-1 / n_samples) * np.sum(loss) + l2_reg
        
    #     elif self.mode_ == 'multinomial':
    #         cost = (-1 / n_samples) * np.sum(y * np.log(h + epsilon)) + l2_reg

    #     return float(cost)


    def _calculate_balanced_weights(self, y: np.ndarray) -> Optional[Dict[Any, float]]:
        """Calculates weights for 'balanced' mode (binary only)."""
        n_samples = len(y)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)            
        weights = n_samples / (n_classes * class_counts)
        
        return {cls: weight for cls, weight in zip(unique_classes, weights)}


    def _one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """Converts class labels into a one-hot encoded matrix."""
        y_one_hot = np.zeros((len(y), self._n_classes))
        for i, label in enumerate(y):
            if label in self._class_map:
                 y_one_hot[i, self._class_map[label]] = 1
        return y_one_hot


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the Logistic Regression model using gradient descent.

        Automatically detects if the problem is binary or multi-class based on `y`.

        Args:
            X (np.ndarray): Training features (n_samples, n_features).
            y (np.ndarray): Training target labels (n_samples,).
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self._n_classes = len(self.classes_)
        self._class_map = {label: i for i, label in enumerate(self.classes_)}

        if self._n_classes < 2:
            raise ValueError("Logistic Regression requires at least 2 classes.")
        
        # --- Mode Detection ---
        if self._n_classes == 2:
            self.mode_ = 'binary'
            
            # Ensure binary labels are 0 and 1 for internal calculations
            # Store the original classes in self.classes_
            binary_y = np.array([1 if label == self.classes_[1] else 0 for label in y])

            # Initialize weights and bias for binary
            self.weights = np.zeros(n_features)
            self.bias = 0.0

            # Handle class weights for binary mode
            self._actual_class_weights = None # Reset
            if self.class_weight_config == 'balanced':
                # Map balanced weights to 0/1 labels
                raw_balanced_weights = self._calculate_balanced_weights(y) # Use original y
                if raw_balanced_weights:
                     self._actual_class_weights = {
                         0: raw_balanced_weights.get(self.classes_[0], 1.0),
                         1: raw_balanced_weights.get(self.classes_[1], 1.0)
                     }
            elif isinstance(self.class_weight_config, dict):
                 # Map provided weights to 0/1 labels
                 self._actual_class_weights = {
                     0: self.class_weight_config.get(self.classes_[0], 1.0),
                     1: self.class_weight_config.get(self.classes_[1], 1.0)
                 }

        else: # Multi-class case
            self.mode_ = 'multinomial'
            
            # Warn if class weights were provided for multi-class
            if self.class_weight_config is not None:
                print("Warning: class_weight parameter is ignored for multi-class classification.")
                self._actual_class_weights = None # Ensure it's not used

            # One-hot encode the target variable
            y_one_hot = self._one_hot_encode(y)
            
            # Initialize weights and bias for multi-class
            self.weights = np.zeros((n_features, self._n_classes))
            self.bias = np.zeros(self._n_classes) # Bias vector, one per class

        # --- Gradient Descent Loop ---
        for i in range(self.n_iterations):
            
            if self.mode_ == 'binary':
                # Binary classification update
                linear_model = np.dot(X, self.weights) + self.bias
                h = self._sigmoid(linear_model) # Predicted probability of class 1
                
                error = h - binary_y # Error against 0/1 labels
                
                # Calculate gradients for binary case
                if self._actual_class_weights is not None:
                    weight_vector = np.array([self._actual_class_weights.get(int(label), 1.0) for label in binary_y])
                    dw = (1 / n_samples) * np.dot(X.T, weight_vector * error) + (self.reg_lambda / n_samples) * self.weights
                    db = (1 / n_samples) * np.sum(weight_vector * error)
                else:
                    dw = (1 / n_samples) * np.dot(X.T, error) + (self.reg_lambda / n_samples) * self.weights
                    db = (1 / n_samples) * np.sum(error)
                
                # Update weights and scalar bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            else: # Multinomial classification update
                linear_model = np.dot(X, self.weights) + self.bias # Z = XW + b
                h = self._softmax(linear_model) # Predicted probabilities (n_samples, n_classes)
                
                error = h - y_one_hot # Error matrix (n_samples, n_classes)
                
                # Calculate gradients for multinomial case
                dw = (1 / n_samples) * np.dot(X.T, error) + (self.reg_lambda / n_samples) * self.weights
                db = (1 / n_samples) * np.sum(error, axis=0) # Sum errors per class for bias grad
                
                # Update weights matrix and bias vector
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Optional: Compute and print loss periodically
            # if (i % 100 == 0):
            #     target_for_loss = binary_y if self.mode_ == 'binary' else y_one_hot
            #     loss = self._compute_loss(h, target_for_loss, self.weights, self.bias, n_samples)
            #     print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for the input data.

        Args:
            X (np.ndarray): Data for which to predict probabilities (n_samples, n_features).

        Returns:
            np.ndarray: Array of predicted probabilities for each class (n_samples, n_classes).
                        Columns correspond to the order of classes in `self.classes_`.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.weights is None or self.bias is None or self.mode_ is None or self._n_classes is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        if self.mode_ == 'binary':
            linear_model = np.dot(X, self.weights) + self.bias
            proba_class1 = self._sigmoid(linear_model)
            proba_class0 = 1 - proba_class1
            return np.column_stack((proba_class0, proba_class1))
            
        else: 
            linear_model = np.dot(X, self.weights) + self.bias
            return self._softmax(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predicts class labels for the input data.

        For binary classification, uses the specified threshold on the probability 
        of the second class (self.classes_[1]). 
        For multi-class classification, predicts the class with the highest probability.

        Args:
            X (np.ndarray): Data for which to predict labels (n_samples, n_features).
            threshold (float, optional): Decision threshold for binary classification. 
                                       Ignored for multi-class. Defaults to 0.5.

        Returns:
            np.ndarray: Predicted class labels (n_samples,). Labels are from `self.classes_`.
            
        Raises:
            ValueError: If the model has not been fitted yet or classes are not defined.
        """
        if self.classes_ is None or self.mode_ is None:
             raise ValueError("Model not fitted or mode/classes not defined.")
             
        probabilities = self.predict_proba(X)
        
        if self.mode_ == 'binary':
             # Use threshold on probability of the second class
             predicted_indices = (probabilities[:, 1] >= threshold).astype(int)
        else: # Multinomial
             # Get the index of the class with the highest probability
             predicted_indices = np.argmax(probabilities, axis=1)
             
        # Map indices (0 or 1 for binary, 0..k-1 for multi-class) back to original class labels
        return self.classes_[predicted_indices]

    def print_coefficients(self) -> None:
        """Prints the learned coefficients (weights) and bias(es)."""
        if self.weights is None or self.bias is None or self.classes_ is None or self.mode_ is None:
            print("Model not fitted yet.")
            return

        print(f"Logistic Regression Coefficients (Mode: {self.mode_}):")
        if self.mode_ == 'binary':
            print("  Class:", self.classes_[1]) # Assuming second class is the positive one
            for idx, coef in enumerate(self.weights):
                print(f"    Feature {idx}: {coef:.4f}")
            print(f"  Bias: {self.bias:.4f}")
        else: # Multinomial
            for i, cls in enumerate(self.classes_):
                print(f"  Class '{cls}':")
                for j, coef in enumerate(self.weights[:, i]):
                    print(f"    Feature {j}: {coef:.4f}")
                # Print corresponding bias for this class
                print(f"    Bias: {self.bias[i]:.4f}") 
        print()

class LDA:
    """
    Implements Linear Discriminant Analysis (LDA) for classification.

    Assumes features follow a Gaussian distribution with a shared covariance matrix 
    across classes.

    Attributes:
        means (Optional[np.ndarray]): Mean vector for each class (shape [n_classes, n_features]).
        shared_covariance (Optional[np.ndarray]): Shared covariance matrix across classes 
                                                  (shape [n_features, n_features]).
        priors (Optional[np.ndarray]): Prior probability for each class (shape [n_classes]).
        classes_ (Optional[np.ndarray]): Unique class labels encountered during fit (shape [n_classes]).
    """
    def __init__(self) -> None:
        """Initializes the LDA classifier."""
        self.means: Optional[np.ndarray] = None
        self.shared_covariance: Optional[np.ndarray] = None # Renamed from cov
        self.priors: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self._n_classes: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the LDA model by estimating class means, priors, and the shared covariance matrix.

        Args:
            X (np.ndarray): Training features (n_samples, n_features).
            y (np.ndarray): Training target labels (n_samples,).
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self._n_classes = len(self.classes_)

        # Initialize storage
        self.means = np.zeros((self._n_classes, n_features))
        self.priors = np.zeros(self._n_classes)
        self.shared_covariance = np.zeros((n_features, n_features))

        # Calculate means and priors per class
        for i, current_class in enumerate(self.classes_):
            # Select samples belonging to the current class
            X_class: np.ndarray = X[y == current_class]
            n_class_samples: int = X_class.shape[0]
            
            if n_class_samples == 0:
                print(f"Warning: Class '{current_class}' has no samples.")
                # Handle appropriately: maybe skip, set mean to zero, etc.
                # Setting prior to zero might be reasonable.
                self.priors[i] = 0.0
                continue # Skip mean and cov calculation for this class
                
            # Calculate mean for the current class
            self.means[i, :] = np.mean(X_class, axis=0)
            # Calculate prior probability for the current class
            self.priors[i] = n_class_samples / n_samples

            # Accumulate sum of squared differences for shared covariance
            # (X_class - mean_i).T @ (X_class - mean_i)
            diff = X_class - self.means[i, :]
            self.shared_covariance += diff.T @ diff

        # Finalize shared covariance matrix calculation
        # Divide by (n_samples - n_classes) for unbiased estimate
        if n_samples <= self._n_classes:
             # Handle case where divisor is non-positive (e.g., n_samples = n_classes)
             # Using identity matrix or regularized covariance might be alternatives
             print(f"Warning: n_samples ({n_samples}) <= n_classes ({self._n_classes}). "
                   f"Cannot compute unbiased shared covariance. Using empirical covariance.")
             # Use empirical covariance (divide by n_samples) as fallback
             if n_samples > 0:
                 self.shared_covariance /= n_samples
             else: # If n_samples is 0, covariance remains zero
                 pass 
        else:
             self.shared_covariance /= (n_samples - self._n_classes)


    def _multivariate_normal_pdf(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Calculates the probability density function (PDF) of the multivariate normal distribution."""
        n_features = X.shape[1]
        # Use pseudo-inverse for numerical stability if covariance is singular
        try:
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)
            # Check for near-zero determinant
            if np.isclose(cov_det, 0):
                 print("Warning: Covariance matrix determinant is close to zero. Using pseudo-inverse.")
                 cov_inv = np.linalg.pinv(cov)
                 # Need a way to handle determinant calculation for pinv case, e.g., regularization
                 # For simplicity here, we might proceed with pinv but det handling needs care.
                 # A small regularization term added to the diagonal of cov before inv/det is common.
                 # cov_det = np.linalg.det(cov + np.eye(n_features) * 1e-6) # Example regularization
                 cov_det = np.prod(np.linalg.svd(cov, compute_uv=False)) # More robust way to get det magnitude
                 if np.isclose(cov_det, 0): cov_det = 1e-15 # Prevent log(0) or division by zero


        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix is singular. Using pseudo-inverse.")
            cov_inv = np.linalg.pinv(cov)
            # Calculate determinant based on SVD for pseudo-inverse case
            # Product of non-zero singular values might approximate determinant magnitude
            s = np.linalg.svd(cov, compute_uv=False)
            cov_det = np.prod(s[s > 1e-10]) # Use a threshold for non-zero singular values
            if np.isclose(cov_det, 0): cov_det = 1e-15 # Prevent log(0) or division by zero


        diff = X - mean # Shape (n_samples, n_features)
        
        # Exponent term: -0.5 * (X - mu).T @ cov_inv @ (X - mu)
        # Vectorized calculation: sum along feature dimension of (diff @ cov_inv) * diff
        exponent = -0.5 * np.sum(np.dot(diff, cov_inv) * diff, axis=1)

        # Normalization constant: 1 / ( (2*pi)^(d/2) * sqrt(det(cov)) )
        denominator = ((2 * np.pi) ** (n_features / 2)) * np.sqrt(np.abs(cov_det)) # Use abs(det)
        
        # Avoid division by zero if denominator is zero
        if np.isclose(denominator, 0):
            # Handle appropriately - maybe return zero probability or a small number
            return np.zeros(X.shape[0]) 
            
        norm_const = 1.0 / denominator
        
        # Return PDF values, clipping exponent to avoid underflow for exp
        return norm_const * np.exp(np.clip(exponent, -700, 700))


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts posterior probabilities for each class using Bayes' theorem.

        P(y=c|X) proportional to P(X|y=c) * P(y=c)

        Args:
            X (np.ndarray): Data for which to predict probabilities (n_samples, n_features).

        Returns:
            np.ndarray: Array of predicted posterior probabilities for each class (n_samples, n_classes).
                        Columns correspond to the order of classes in `self.classes_`.
                        
        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.means is None or self.shared_covariance is None or self.priors is None or self._n_classes is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        n_samples = X.shape[0]
        # Store likelihood * prior for each class
        likelihood_times_prior = np.zeros((n_samples, self._n_classes))

        for i in range(self._n_classes):
            # Calculate likelihood P(X|y=c) using multivariate normal PDF
            likelihood = self._multivariate_normal_pdf(X, self.means[i], self.shared_covariance)
            # Multiply by prior P(y=c)
            likelihood_times_prior[:, i] = likelihood * self.priors[i]

        # Normalize to get posterior probabilities P(y=c|X)
        # Sum of (likelihood * prior) over all classes acts as evidence P(X)
        evidence = np.sum(likelihood_times_prior, axis=1, keepdims=True)
        
        # Handle cases where evidence is zero (all likelihoods*priors were zero)
        # Avoid division by zero - assign uniform probability or handle as error
        posterior_proba = np.zeros_like(likelihood_times_prior)
        non_zero_evidence_mask = (evidence > 1e-15).flatten() # Flatten for 1D indexing
        
        if np.any(non_zero_evidence_mask):
            posterior_proba[non_zero_evidence_mask] = likelihood_times_prior[non_zero_evidence_mask] / evidence[non_zero_evidence_mask]
            
        # Optional: Assign uniform probability if evidence is zero
        if np.any(~non_zero_evidence_mask):
            # print("Warning: Zero evidence encountered for some samples. Assigning uniform probability.")
            posterior_proba[~non_zero_evidence_mask] = 1.0 / self._n_classes
            
        return posterior_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the input data by choosing the class with the highest posterior probability.

        Args:
            X (np.ndarray): Data for which to predict labels (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (n_samples,). Labels are from `self.classes_`.
            
        Raises:
            ValueError: If the model has not been fitted yet or classes are not defined.
        """
        if self.classes_ is None:
             raise ValueError("Model not fitted or classes not defined.")
             
        posterior_proba = self.predict_proba(X)
        # Get the index of the class with the highest probability for each sample
        predicted_indices = np.argmax(posterior_proba, axis=1)
        # Map indices back to original class labels
        return self.classes_[predicted_indices]

    def print_coefficients(self) -> None:
        """Prints the estimated class means and priors."""
        if self.means is None or self.priors is None or self.classes_ is None:
            print("Model not fitted yet.")
            return

        print("LDA Parameters:")
        for i, cls in enumerate(self.classes_):
            print(f"  Class '{cls}':")
            print(f"    Prior: {self.priors[i]:.4f}")
            print(f"    Means: {np.array2string(self.means[i], precision=4, suppress_small=True)}")
        # Optionally print shared covariance, but it can be large
        # print("\n  Shared Covariance Matrix:")
        # print(np.array2string(self.shared_covariance, precision=4, suppress_small=True))
        print()

DecisionTreeNode = Dict[str, Any] 

class DecisionTree:
    """
    Implements a Decision Tree classifier using the CART algorithm (Entropy/Information Gain).

    Attributes:
        max_depth (Optional[int]): Maximum depth the tree is allowed to grow. 
            If None, nodes are expanded until all leaves are pure or contain 
            fewer samples than min_samples_split.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        tree (Optional[DecisionTreeNode]): The learned decision tree structure, represented 
            as a nested dictionary, after fitting.
        feature_importances_ (Optional[np.ndarray]): Importance of each feature computed as the 
            (normalized) total reduction of the criterion brought by that feature. Initialized after fit.
    """
    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2) -> None:
        """
        Initializes the DecisionTree classifier.

        Args:
            max_depth (Optional[int], optional): Maximum depth of the tree. Defaults to None (no limit).
            min_samples_split (int, optional): Minimum samples needed to split a node. Defaults to 2.
        """
        self.max_depth: Optional[int] = max_depth
        self.min_samples_split: int = min_samples_split
        self.tree: Optional[DecisionTreeNode] = None
        self._n_features: Optional[int] = None # Store number of features seen during fit
        self.feature_importances_: Optional[np.ndarray] = None


    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Calculates the Shannon entropy for a set of labels."""
        n_samples = len(y)
        if n_samples <= 1: # Entropy is 0 for a single sample or empty set
            return 0.0
            
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n_samples
        # Use log2 for entropy calculation, add epsilon for numerical stability
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15)) 
        return float(entropy) # Ensure float return type

    def _calculate_information_gain(
        self, 
        X_column: np.ndarray, 
        y: np.ndarray, 
        threshold: float
    ) -> float:
        """Calculates the Information Gain for a potential split."""
        
        # Calculate parent entropy (entropy before split)
        parent_entropy = self._calculate_entropy(y)

        # Split data based on threshold
        left_mask = (X_column <= threshold)
        right_mask = ~left_mask # Equivalent to > threshold

        y_left = y[left_mask]
        y_right = y[right_mask]

        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        # If either split is empty, gain is 0
        if n_left == 0 or n_right == 0:
            return 0.0

        # Calculate weighted average entropy of children
        entropy_left = self._calculate_entropy(y_left)
        entropy_right = self._calculate_entropy(y_right)
        child_entropy = (n_left / n_total) * entropy_left + (n_right / n_total) * entropy_right

        # Information Gain = Parent Entropy - Weighted Child Entropy
        information_gain = parent_entropy - child_entropy
        return float(information_gain) # Ensure float return type

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Finds the best feature and threshold for splitting the data."""
        n_samples, n_features = X.shape
        best_gain = -1.0 # Initialize with a value lower than any possible gain
        best_feature_index: Optional[int] = None
        best_threshold: Optional[float] = None
        
        # Ensure there's potential for a split
        if n_samples < self.min_samples_split or len(np.unique(y)) <= 1:
            return best_feature_index, best_threshold, best_gain

        # Iterate through each feature
        for feature_index in range(n_features):
            X_column = X[:, feature_index]
            # Consider unique values in the feature as potential thresholds
            # Sorting thresholds might make processing slightly more structured
            # but unique() is sufficient.
            potential_thresholds = np.unique(X_column)
            
            # Often, thresholds are considered midpoints between unique sorted values
            # For simplicity here, we use unique values directly.
            # midpoints = (potential_thresholds[:-1] + potential_thresholds[1:]) / 2

            for threshold in potential_thresholds: # Or iterate through midpoints
                # Calculate information gain for this split
                gain = self._calculate_information_gain(X_column, y, threshold)

                # Update best split if current gain is higher
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = float(threshold) # Ensure threshold is float

        return best_feature_index, best_threshold, best_gain

    def _build_tree_recursive(self, X: np.ndarray, y: np.ndarray, current_depth: int) -> DecisionTreeNode:
        """Recursively builds the decision tree."""
        n_samples, n_features = X.shape
        
        # --- Stopping Criteria ---
        # 1. Max depth reached
        # 2. Minimum samples for split not met
        # 3. Node is pure (only one class left)
        is_max_depth = (self.max_depth is not None and current_depth >= self.max_depth)
        is_min_samples = (n_samples < self.min_samples_split)
        is_pure = (len(np.unique(y)) == 1)

        if is_max_depth or is_min_samples or is_pure:
            # Create a leaf node: predict the most common class
            leaf_value = _mode_1d(y) # Find most frequent class
            return {'is_leaf': True, 'class': leaf_value, 'n_samples': n_samples, 'entropy': self._calculate_entropy(y)}

        # --- Find the best split ---
        feature_index, threshold, gain = self._find_best_split(X, y)

        # --- Additional Stopping Criterion: No beneficial split found ---
        if gain <= 0 or feature_index is None or threshold is None:
             leaf_value = _mode_1d(y)
             # Update feature importance contribution even if stopping early
             # (Could argue gain=0 means no contribution, depends on definition)
             return {'is_leaf': True, 'class': leaf_value, 'n_samples': n_samples, 'entropy': self._calculate_entropy(y)}
             
        # --- Update Feature Importance ---
        # Contribution is gain * number of samples affected
        if self.feature_importances_ is not None:
            self.feature_importances_[feature_index] += gain * n_samples


        # --- Split the data ---
        left_mask = (X[:, feature_index] <= threshold)
        right_mask = ~left_mask

        X_left, y_left = X[left_mask, :], y[left_mask]
        X_right, y_right = X[right_mask, :], y[right_mask]

        # --- Recursively build subtrees ---
        left_subtree = self._build_tree_recursive(X_left, y_left, current_depth + 1)
        right_subtree = self._build_tree_recursive(X_right, y_right, current_depth + 1)

        # --- Create internal node ---
        return {
            'is_leaf': False,
            'feature_index': feature_index,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree,
            'n_samples': n_samples,
            'entropy': self._calculate_entropy(y), # Store entropy before split
            'gain': gain # Store gain achieved by this split
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Builds the decision tree classifier from the training set (X, y).

        Args:
            X (np.ndarray): Training features (n_samples, n_features).
            y (np.ndarray): Training target labels (n_samples,).
        """
        self._n_features = X.shape[1]
        # Initialize feature importances (unnormalized)
        self.feature_importances_ = np.zeros(self._n_features) 
        
        self.tree = self._build_tree_recursive(X, y, current_depth=0)
        
        # Normalize feature importances
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance
        else:
            # Handle case where no splits occurred or all gains were zero
            self.feature_importances_ = np.zeros(self._n_features)


    def _predict_single_sample(self, sample: np.ndarray, node: DecisionTreeNode) -> Any:
        """Predicts the class label for a single sample by traversing the tree."""
        # If it's a leaf node, return its class prediction
        if node['is_leaf']:
            return node['class']

        # If it's an internal node, decide whether to go left or right
        feature_index = node['feature_index']
        threshold = node['threshold']
        
        if sample[feature_index] <= threshold:
            # Recursively predict using the left subtree
            return self._predict_single_sample(sample, node['left'])
        else:
            # Recursively predict using the right subtree
            return self._predict_single_sample(sample, node['right'])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for multiple samples.

        Args:
            X (np.ndarray): Data for which to predict labels (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (n_samples,).
            
        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.tree is None:
            raise ValueError("Decision tree has not been fitted yet. Call fit() first.")
            
        # Apply the prediction logic to each sample (row) in X
        predictions = np.array([self._predict_single_sample(sample, self.tree) for sample in X])
        return predictions
    
    def _print_tree_recursive(self, node: DecisionTreeNode, depth: int = 0, feature_names: Optional[List[str]] = None) -> None:
        """Helper function to recursively print the tree structure."""
        indent = "  " * depth
        
        if node['is_leaf']:
            print(f"{indent}Leaf: Predict Class={node['class']} (Samples={node['n_samples']}, Entropy={node['entropy']:.3f})")
            return

        feature_idx = node['feature_index']
        threshold = node['threshold']
        feature_display = f"Feature_{feature_idx}"
        if feature_names and feature_idx < len(feature_names):
             feature_display = feature_names[feature_idx]
             
        print(f"{indent}Node: If {feature_display} <= {threshold:.3f} (Samples={node['n_samples']}, Entropy={node['entropy']:.3f}, Gain={node['gain']:.3f}):")
        
        # Print left subtree
        self._print_tree_recursive(node['left'], depth + 1, feature_names)
        
        # Print right subtree
        print(f"{indent}Node: Else (If {feature_display} > {threshold:.3f}):")
        self._print_tree_recursive(node['right'], depth + 1, feature_names)


    def print_tree(self, feature_names: Optional[List[str]] = None) -> None:
        """
        Prints a textual representation of the learned decision tree.

        Args:
            feature_names (Optional[List[str]], optional): List of names for the features, 
                corresponding to their column indices. If provided, names are used 
                in the output instead of indices. Defaults to None.
        """
        if self.tree is None:
            print("Tree not fitted yet.")
            return
        if feature_names and self._n_features != len(feature_names):
             print(f"Warning: Number of feature names ({len(feature_names)}) does not match number of features ({self._n_features}). Using indices.")
             feature_names = None # Reset if mismatch
             
        self._print_tree_recursive(self.tree, feature_names=feature_names)


class RandomForest:
    """
    Implements a Random Forest classifier using an ensemble of Decision Trees.

    Each tree is built on a bootstrap sample of the data, and splits consider
    only a random subset of features. Predictions are made by aggregating 
    the predictions of individual trees (majority vote for classification).

    Attributes:
        n_estimators (int): The number of decision trees in the forest.
        max_depth (Optional[int]): Maximum depth allowed for individual trees.
        min_samples_split (int): Minimum number of samples required to split a node 
            in individual trees.
        max_features (Optional[Union[int, float, str]]): The number of features to consider 
            when looking for the best split in individual trees.
            - If int: Use max_features features.
            - If float: Use max_features * n_features features (rounded down).
            - If 'sqrt': Use sqrt(n_features) features.
            - If None: Use sqrt(n_features). Defaults to None.
        random_state (Optional[int]): Seed for the random number generator for 
            reproducibility of bootstrapping and feature selection.
        trees (List[DecisionTree]): List containing the individual decision trees 
            trained in the forest.
        feature_indices_ (List[np.ndarray]): List containing the indices of the features 
            used by each corresponding tree in `self.trees`.
        classes_ (Optional[np.ndarray]): Unique class labels encountered during fit.
        n_classes_ (Optional[int]): The number of unique classes.
        feature_importances_ (Optional[np.ndarray]): Importance of each feature, averaged 
            over all trees in the forest. Computed after fit.
    """
    def __init__(
        self, 
        n_estimators: int = 100, 
        max_depth: Optional[int] = None, 
        min_samples_split: int = 2, 
        max_features: Optional[Union[int, float, str]] = None, 
        random_state: Optional[int] = None
    ):
        """
        Initializes the RandomForest classifier.

        Args:
            n_estimators (int, optional): Number of trees in the forest. Defaults to 100.
            max_depth (Optional[int], optional): Max depth of individual trees. Defaults to None.
            min_samples_split (int, optional): Min samples to split a node in trees. Defaults to 2.
            max_features (Optional[Union[int, float, str]], optional): Number/proportion of 
                features to consider per split ('sqrt' or None use sqrt(n_features)). Defaults to None.
            random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.
        """
        self.n_estimators: int = n_estimators
        self.max_depth: Optional[int] = max_depth
        self.min_samples_split: int = min_samples_split
        self.max_features_config: Optional[Union[int, float, str]] = max_features # Store original config
        self._max_features_internal: Optional[int] = None # Actual number calculated in fit
        self.random_state: Optional[int] = random_state
        self.trees: List[DecisionTree] = []
        self.feature_indices_: List[np.ndarray] = [] # Store indices used by each tree
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None


    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a bootstrap sample (sampling with replacement)."""
        n_samples = X.shape[0]
        # Generate random indices with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _calculate_max_features(self, n_features: int) -> int:
        """Determines the number of features to use per split based on config."""
        if self.max_features_config is None or self.max_features_config == 'sqrt':
            return max(1, int(np.sqrt(n_features))) # Ensure at least 1 feature
        elif isinstance(self.max_features_config, float):
            # Ensure float is between 0.0 and 1.0
            if 0.0 < self.max_features_config <= 1.0:
                 return max(1, int(self.max_features_config * n_features))
            else:
                 print(f"Warning: max_features float ({self.max_features_config}) out of range (0, 1]. Using 'sqrt'.")
                 return max(1, int(np.sqrt(n_features)))
        elif isinstance(self.max_features_config, int):
            # Ensure int is positive and not more than n_features
            if 0 < self.max_features_config <= n_features:
                return self.max_features_config
            else:
                 print(f"Warning: max_features int ({self.max_features_config}) out of range (1, {n_features}]. Using 'sqrt'.")
                 return max(1, int(np.sqrt(n_features)))
        else: # Handle unexpected types
            print(f"Warning: Invalid max_features type ({type(self.max_features_config)}). Using 'sqrt'.")
            return max(1, int(np.sqrt(n_features)))


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Builds the forest of decision trees from the training set (X, y).

        Args:
            X (np.ndarray): Training features (n_samples, n_features).
            y (np.ndarray): Training target labels (n_samples,).
        """
        # Set random seed for reproducibility if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.trees = []
        self.feature_indices_ = []
        self.feature_importances_ = np.zeros(n_features) # Initialize importances
        
        # Determine the actual number of features to use per split
        self._max_features_internal = self._calculate_max_features(n_features)

        # Build each tree in the forest
        for i in range(self.n_estimators):
            # 1. Create bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # 2. Select random subset of features for this tree
            # Ensure max_features is not larger than n_features
            n_features_to_select = min(self._max_features_internal, n_features) 
            current_feature_indices = np.random.choice(
                n_features, size=n_features_to_select, replace=False
            )
            self.feature_indices_.append(current_feature_indices)

            # 3. Train a Decision Tree on the bootstrap sample and selected features
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            # Fit the tree using only the selected features
            tree.fit(X_sample[:, current_feature_indices], y_sample)
            
            # 4. Store the trained tree
            self.trees.append(tree)
            
            # 5. Accumulate feature importances (scaled by tree's contribution)
            if tree.feature_importances_ is not None:
                 # Map tree's feature importances back to original feature indices
                 self.feature_importances_[current_feature_indices] += tree.feature_importances_

        # Average feature importances over all trees
        if self.n_estimators > 0:
             self.feature_importances_ /= self.n_estimators
        else:
             self.feature_importances_ = np.zeros(n_features) # No trees, no importance


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for the input data by averaging tree predictions.

        Args:
            X (np.ndarray): Data for which to predict probabilities (n_samples, n_features).

        Returns:
            np.ndarray: Array of predicted probabilities for each class (n_samples, n_classes_).
                        Columns correspond to the order of classes in `self.classes_`.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.trees or self.classes_ is None or self.n_classes_ is None:
            raise ValueError("Random Forest model not fitted yet. Call fit() first.")

        n_samples = X.shape[0]
        # Initialize array to store aggregated probabilities or counts
        proba_sum = np.zeros((n_samples, self.n_classes_))

        # Get predictions from each tree
        for tree, feature_idx in zip(self.trees, self.feature_indices_):
            # Predict using only the features the tree was trained on
            tree_predictions = tree.predict(X[:, feature_idx])
            
            # Add probabilities (as counts) to the sum
            # This loop can be vectorized further if needed
            for i in range(n_samples):
                predicted_class = tree_predictions[i]
                # Find the index corresponding to the predicted class
                class_index = np.where(self.classes_ == predicted_class)[0]
                if len(class_index) > 0: # Ensure class was seen during fit
                    proba_sum[i, class_index[0]] += 1

        # Normalize counts to get probabilities
        # Avoid division by zero if a sample got zero predictions (unlikely with many trees)
        total_predictions = proba_sum.sum(axis=1, keepdims=True)
        # Replace 0 sums with 1 to avoid NaN, probability will be 0 anyway
        total_predictions[total_predictions == 0] = 1 
        
        probabilities = proba_sum / total_predictions
        
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the input data using majority vote among trees.

        Args:
            X (np.ndarray): Data for which to predict labels (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels (n_samples,). Labels are from `self.classes_`.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.trees or self.classes_ is None:
            raise ValueError("Random Forest model not fitted yet. Call fit() first.")
            
        n_samples = X.shape[0]
        # Store predictions from all trees (n_samples, n_estimators)
        all_predictions = np.zeros((n_samples, self.n_estimators), dtype=object) # Use object dtype for mixed label types

        for i, (tree, feature_idx) in enumerate(zip(self.trees, self.feature_indices_)):
             all_predictions[:, i] = tree.predict(X[:, feature_idx])

        # Find the most frequent prediction (mode) for each sample
        # Apply the NumPy mode function along the axis of estimators (axis=1)
        final_predictions = np.apply_along_axis(_mode_1d, axis=1, arr=all_predictions)

        return final_predictions

    # Optional: Add print_coefficients-like method if needed, maybe print average feature importances
    def print_feature_importances(self, feature_names: Optional[List[str]] = None) -> None:
         """Prints the calculated feature importances."""
         if self.feature_importances_ is None:
             print("Model not fitted yet or feature importances not computed.")
             return
         
         print("Random Forest Feature Importances:")
         indices = np.argsort(self.feature_importances_)[::-1] # Sort descending
         
         n_features = len(self.feature_importances_)
         if feature_names and len(feature_names) != n_features:
             print(f"Warning: Number of feature names ({len(feature_names)}) does not match number of features ({n_features}). Using indices.")
             feature_names = None
             
         for i in range(n_features):
             idx = indices[i]
             name = feature_names[idx] if feature_names else f"Feature {idx}"
             print(f"  {name}: {self.feature_importances_[idx]:.4f}")
         print()
