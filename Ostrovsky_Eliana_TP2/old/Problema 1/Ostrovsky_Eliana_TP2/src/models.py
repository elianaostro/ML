import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any

def _mode_1d(a: np.ndarray) -> Union[int, float, Any]:
    """Finds the most frequent value in a 1D NumPy array."""
    vals, counts = np.unique(a, return_counts=True)
    return vals[np.argmax(counts)]

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
        
        if self._n_classes == 2:
            self.mode_ = 'binary'
            
            binary_y = np.array([1 if label == self.classes_[1] else 0 for label in y])

            self.weights = np.zeros(n_features)
            self.bias = 0.0

            self._actual_class_weights = None
            if self.class_weight_config == 'balanced':
                raw_balanced_weights = self._calculate_balanced_weights(y)
                if raw_balanced_weights:
                    self._actual_class_weights = {
                        0: raw_balanced_weights.get(self.classes_[0], 1.0),
                        1: raw_balanced_weights.get(self.classes_[1], 1.0)
                    }
            elif isinstance(self.class_weight_config, dict):
                self._actual_class_weights = {
                    0: self.class_weight_config.get(self.classes_[0], 1.0),
                    1: self.class_weight_config.get(self.classes_[1], 1.0)
                }

        else:
            self.mode_ = 'multinomial'
            
            if self.class_weight_config is not None:
                print("Warning: class_weight parameter is ignored for multi-class classification.")
                self._actual_class_weights = None

            y_one_hot = self._one_hot_encode(y)
            
            self.weights = np.zeros((n_features, self._n_classes))
            self.bias = np.zeros(self._n_classes)

        for i in range(self.n_iterations):
            
            if self.mode_ == 'binary':
                linear_model = np.dot(X, self.weights) + self.bias
                h = self._sigmoid(linear_model)
                
                error = h - binary_y
                
                if self._actual_class_weights is not None:
                    weight_vector = np.array([self._actual_class_weights.get(int(label), 1.0) for label in binary_y])
                    dw_loss = (1 / n_samples) * np.dot(X.T, weight_vector * error)
                    dw_reg = (self.reg_lambda / n_samples) * self.weights
                    dw = dw_loss + dw_reg
                    db = (1 / n_samples) * np.sum(weight_vector * error)
                else:
                    dw_loss = (1 / n_samples) * np.dot(X.T, error)
                    dw_reg = (self.reg_lambda / n_samples) * self.weights
                    dw = dw_loss + dw_reg
                    db = (1 / n_samples) * np.sum(error)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            else:
                linear_model = np.dot(X, self.weights) + self.bias
                h = self._softmax(linear_model)
                
                error = h - y_one_hot
                
                dw_loss = (1 / n_samples) * np.dot(X.T, error)
                dw_reg = (self.reg_lambda / n_samples) * self.weights
                dw = dw_loss + dw_reg
                db = (1 / n_samples) * np.sum(error, axis=0)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

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
            predicted_indices = (probabilities[:, 1] >= threshold).astype(int)
        else:
            predicted_indices = np.argmax(probabilities, axis=1)
             
        return self.classes_[predicted_indices]

    def print_coefficients(self) -> None:
        """Prints the learned coefficients (weights) and bias(es)."""
        if self.weights is None or self.bias is None or self.classes_ is None or self.mode_ is None:
            print("Model not fitted yet.")
            return

        print(f"Logistic Regression Coefficients (Mode: {self.mode_}):")
        if self.mode_ == 'binary':
            print("  Class:", self.classes_[1])
            for idx, coef in enumerate(self.weights):
                print(f"    Feature {idx}: {coef:.4f}")
            print(f"  Bias: {self.bias:.4f}")
        else:
            for i, cls in enumerate(self.classes_):
                print(f"  Class '{cls}':")
                for j, coef in enumerate(self.weights[:, i]):
                    print(f"    Feature {j}: {coef:.4f}")
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
        self.shared_covariance: Optional[np.ndarray] = None
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

        self.means = np.zeros((self._n_classes, n_features))
        self.priors = np.zeros(self._n_classes)
        self.shared_covariance = np.zeros((n_features, n_features))

        for i, current_class in enumerate(self.classes_):
            X_class: np.ndarray = X[y == current_class]
            n_class_samples: int = X_class.shape[0]
            
            if n_class_samples == 0:
                print(f"Warning: Class '{current_class}' has no samples.")
                self.priors[i] = 0.0
                continue
                
            self.means[i, :] = np.mean(X_class, axis=0)
            self.priors[i] = n_class_samples / n_samples

            diff = X_class - self.means[i, :]
            self.shared_covariance += diff.T @ diff

        if n_samples <= self._n_classes:
             print(f"Warning: n_samples ({n_samples}) <= n_classes ({self._n_classes}). "
                   f"Cannot compute unbiased shared covariance. Using empirical covariance.")
             if n_samples > 0:
                 self.shared_covariance /= n_samples
             else:
                 pass 
        else:
             self.shared_covariance /= (n_samples - self._n_classes)


    def _multivariate_normal_pdf(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Calculates the probability density function (PDF) of the multivariate normal distribution."""
        n_features = X.shape[1]
        try:
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)
            if np.isclose(cov_det, 0):
                 print("Warning: Covariance matrix determinant is close to zero. Using pseudo-inverse.")
                 cov_inv = np.linalg.pinv(cov)
                 cov_det = np.prod(np.linalg.svd(cov, compute_uv=False))
                 if np.isclose(cov_det, 0): cov_det = 1e-15


        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix is singular. Using pseudo-inverse.")
            cov_inv = np.linalg.pinv(cov)
            s = np.linalg.svd(cov, compute_uv=False)
            cov_det = np.prod(s[s > 1e-10])
            if np.isclose(cov_det, 0): cov_det = 1e-15


        diff = X - mean
        
        exponent = -0.5 * np.sum(np.dot(diff, cov_inv) * diff, axis=1)

        denominator = ((2 * np.pi) ** (n_features / 2)) * np.sqrt(np.abs(cov_det))
        
        if np.isclose(denominator, 0):
            return np.zeros(X.shape[0]) 
            
        norm_const = 1.0 / denominator
        
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
        likelihood_times_prior = np.zeros((n_samples, self._n_classes))

        for i in range(self._n_classes):
            likelihood = self._multivariate_normal_pdf(X, self.means[i], self.shared_covariance)
            likelihood_times_prior[:, i] = likelihood * self.priors[i]

        evidence = np.sum(likelihood_times_prior, axis=1, keepdims=True)
        
        posterior_proba = np.zeros_like(likelihood_times_prior)
        non_zero_evidence_mask = (evidence > 1e-15).flatten()
        
        if np.any(non_zero_evidence_mask):
            posterior_proba[non_zero_evidence_mask] = likelihood_times_prior[non_zero_evidence_mask] / evidence[non_zero_evidence_mask]
            
        if np.any(~non_zero_evidence_mask):
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
        predicted_indices = np.argmax(posterior_proba, axis=1)
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
        self._n_features: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None


    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Calculates the Shannon entropy for a set of labels."""
        n_samples = len(y)
        if n_samples <= 1:
            return 0.0
            
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n_samples
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15)) 
        return float(entropy)

    def _calculate_information_gain( self, X_column: np.ndarray, y: np.ndarray, threshold: float) -> float:
        """Calculates the Information Gain for a potential split."""
        
        parent_entropy = self._calculate_entropy(y)

        left_mask = (X_column <= threshold)
        right_mask = ~left_mask

        y_left = y[left_mask]
        y_right = y[right_mask]

        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        if n_left == 0 or n_right == 0:
            return 0.0

        entropy_left = self._calculate_entropy(y_left)
        entropy_right = self._calculate_entropy(y_right)
        child_entropy = (n_left / n_total) * entropy_left + (n_right / n_total) * entropy_right

        information_gain = parent_entropy - child_entropy
        return float(information_gain)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Finds the best feature and threshold for splitting the data."""
        n_samples, n_features = X.shape
        best_gain = -1.0
        best_feature_index: Optional[int] = None
        best_threshold: Optional[float] = None
        
        if n_samples < self.min_samples_split or len(np.unique(y)) <= 1:
            return best_feature_index, best_threshold, best_gain

        for feature_index in range(n_features):
            X_column = X[:, feature_index]
            potential_thresholds = np.unique(X_column)

            for threshold in potential_thresholds:
                gain = self._calculate_information_gain(X_column, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = float(threshold)

        return best_feature_index, best_threshold, best_gain

    def _build_tree_recursive(self, X: np.ndarray, y: np.ndarray, current_depth: int) -> DecisionTreeNode:
        """Recursively builds the decision tree."""
        n_samples, n_features = X.shape
        
        is_max_depth = (self.max_depth is not None and current_depth >= self.max_depth)
        is_min_samples = (n_samples < self.min_samples_split)
        is_pure = (len(np.unique(y)) == 1)

        if is_max_depth or is_min_samples or is_pure:
            leaf_value = _mode_1d(y)
            return {'is_leaf': True, 'class': leaf_value, 'n_samples': n_samples, 'entropy': self._calculate_entropy(y)}

        feature_index, threshold, gain = self._find_best_split(X, y)

        if gain <= 0 or feature_index is None or threshold is None:
             leaf_value = _mode_1d(y)
             return {'is_leaf': True, 'class': leaf_value, 'n_samples': n_samples, 'entropy': self._calculate_entropy(y)}
             
        if self.feature_importances_ is not None:
            self.feature_importances_[feature_index] += gain * n_samples


        left_mask = (X[:, feature_index] <= threshold)
        right_mask = ~left_mask

        X_left, y_left = X[left_mask, :], y[left_mask]
        X_right, y_right = X[right_mask, :], y[right_mask]

        left_subtree = self._build_tree_recursive(X_left, y_left, current_depth + 1)
        right_subtree = self._build_tree_recursive(X_right, y_right, current_depth + 1)

        return {
            'is_leaf': False,
            'feature_index': feature_index,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree,
            'n_samples': n_samples,
            'entropy': self._calculate_entropy(y),
            'gain': gain
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Builds the decision tree classifier from the training set (X, y).

        Args:
            X (np.ndarray): Training features (n_samples, n_features).
            y (np.ndarray): Training target labels (n_samples,).
        """
        self._n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self._n_features) 
        
        self.tree = self._build_tree_recursive(X, y, current_depth=0)
        
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance
        else:
            self.feature_importances_ = np.zeros(self._n_features)


    def _predict_single_sample(self, sample: np.ndarray, node: DecisionTreeNode) -> Any:
        """Predicts the class label for a single sample by traversing the tree."""
        if node['is_leaf']:
            return node['class']

        feature_index = node['feature_index']
        threshold = node['threshold']
        
        if sample[feature_index] <= threshold:
            return self._predict_single_sample(sample, node['left'])
        else:
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
        
        self._print_tree_recursive(node['left'], depth + 1, feature_names)
        
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
             feature_names = None
             
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
    def __init__( self, n_estimators: int = 100, max_depth: Optional[int] = None, min_samples_split: int = 2, 
                 max_features: Optional[Union[int, float, str]] = None, random_state: Optional[int] = None):
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
        self.max_features_config: Optional[Union[int, float, str]] = max_features
        self._max_features_internal: Optional[int] = None
        self.random_state: Optional[int] = random_state
        self.trees: List[DecisionTree] = []
        self.feature_indices_: List[np.ndarray] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None


    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a bootstrap sample (sampling with replacement)."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _calculate_max_features(self, n_features: int) -> int:
        """Determines the number of features to use per split based on config."""
        if self.max_features_config is None or self.max_features_config == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif isinstance(self.max_features_config, float):
            if 0.0 < self.max_features_config <= 1.0:
                 return max(1, int(self.max_features_config * n_features))
            else:
                 print(f"Warning: max_features float ({self.max_features_config}) out of range (0, 1]. Using 'sqrt'.")
                 return max(1, int(np.sqrt(n_features)))
        elif isinstance(self.max_features_config, int):
            if 0 < self.max_features_config <= n_features:
                return self.max_features_config
            else:
                 print(f"Warning: max_features int ({self.max_features_config}) out of range (1, {n_features}]. Using 'sqrt'.")
                 return max(1, int(np.sqrt(n_features)))
        else:
            print(f"Warning: Invalid max_features type ({type(self.max_features_config)}). Using 'sqrt'.")
            return max(1, int(np.sqrt(n_features)))


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Builds the forest of decision trees from the training set (X, y).

        Args:
            X (np.ndarray): Training features (n_samples, n_features).
            y (np.ndarray): Training target labels (n_samples,).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.trees = []
        self.feature_indices_ = []
        self.feature_importances_ = np.zeros(n_features)
        
        self._max_features_internal = self._calculate_max_features(n_features)

        for i in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            n_features_to_select = min(self._max_features_internal, n_features) 
            current_feature_indices = np.random.choice(
                n_features, size=n_features_to_select, replace=False
            )
            self.feature_indices_.append(current_feature_indices)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_sample[:, current_feature_indices], y_sample)
            
            self.trees.append(tree)
            
            if tree.feature_importances_ is not None:
                 self.feature_importances_[current_feature_indices] += tree.feature_importances_

        if self.n_estimators > 0:
             self.feature_importances_ /= self.n_estimators
        else:
             self.feature_importances_ = np.zeros(n_features)


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
        proba_sum = np.zeros((n_samples, self.n_classes_))

        for tree, feature_idx in zip(self.trees, self.feature_indices_):
            tree_predictions = tree.predict(X[:, feature_idx])
            
            for i in range(n_samples):
                predicted_class = tree_predictions[i]
                class_index = np.where(self.classes_ == predicted_class)[0]
                if len(class_index) > 0:
                    proba_sum[i, class_index[0]] += 1

        total_predictions = proba_sum.sum(axis=1, keepdims=True)
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
        all_predictions = np.zeros((n_samples, self.n_estimators), dtype=object)

        for i, (tree, feature_idx) in enumerate(zip(self.trees, self.feature_indices_)):
             all_predictions[:, i] = tree.predict(X[:, feature_idx])

        final_predictions = np.apply_along_axis(_mode_1d, axis=1, arr=all_predictions)

        return final_predictions

    def print_feature_importances(self, feature_names: Optional[List[str]] = None) -> None:
         """Prints the calculated feature importances."""
         if self.feature_importances_ is None:
             print("Model not fitted yet or feature importances not computed.")
             return
         
         print("Random Forest Feature Importances:")
         indices = np.argsort(self.feature_importances_)[::-1]
         
         n_features = len(self.feature_importances_)
         if feature_names and len(feature_names) != n_features:
             print(f"Warning: Number of feature names ({len(feature_names)}) does not match number of features ({n_features}). Using indices.")
             feature_names = None
             
         for i in range(n_features):
             idx = indices[i]
             name = feature_names[idx] if feature_names else f"Feature {idx}"
             print(f"  {name}: {self.feature_importances_[idx]:.4f}")
         print()
