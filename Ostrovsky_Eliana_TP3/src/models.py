import numpy as np
from typing import List, Tuple, Dict, Any

class NeuralNetwork:
    """
    Basic neural network implementation with L hidden layers.
    Uses ReLU activation for hidden layers and softmax for output layer.
    Trained using backpropagation and standard gradient descent with cross-entropy loss.
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        """
        Initialize a neural network with specified layer sizes.
        
        Args:
            layer_sizes: List containing the number of neurons in each layer.
                         First element is input size, last element is output size.
            learning_rate: Learning rate for gradient descent.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(1, self.num_layers):
            # He initialization for ReLU activation
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 
                               np.sqrt(2.0 / layer_sizes[i-1]))
            self.biases.append(np.zeros((1, layer_sizes[i])))
    
    def relu(self, Z: np.ndarray) -> np.ndarray:
        """
        ReLU activation function.
        
        Args:
            Z: Input array.
            
        Returns:
            Output after applying ReLU.
        """
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU activation function.
        
        Args:
            Z: Input array.
            
        Returns:
            Derivative of ReLU for the input.
        """
        return np.where(Z > 0, 1, 0)
    
    def softmax(self, Z: np.ndarray) -> np.ndarray:
        """
        Softmax activation function.
        
        Args:
            Z: Input array.
            
        Returns:
            Output after applying softmax.
        """
        # Shift values for numerical stability
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            X: Input data of shape (n_samples, input_size).
            
        Returns:
            Output predictions of shape (n_samples, output_size).
        """
        self.Z_values = []  # Pre-activation values
        self.A_values = [X]  # Activation values, with input as first activation
        
        # Pass through hidden layers with ReLU activation
        for i in range(self.num_layers - 2):
            Z = np.dot(self.A_values[-1], self.weights[i]) + self.biases[i]
            self.Z_values.append(Z)
            A = self.relu(Z)
            self.A_values.append(A)
        
        # Output layer with softmax activation
        Z = np.dot(self.A_values[-1], self.weights[-1]) + self.biases[-1]
        self.Z_values.append(Z)
        A = self.softmax(Z)
        self.A_values.append(A)
        
        return self.A_values[-1]
    
    def cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate cross-entropy loss.
        
        Args:
            y_true: True labels (indices).
            y_pred: Predicted probabilities.
            
        Returns:
            Cross-entropy loss value.
        """
        m = y_true.shape[0]
        
        # Convert y_true to one-hot encoding
        y_true_one_hot = np.zeros((m, self.layer_sizes[-1]))
        y_true_one_hot[np.arange(m), y_true.astype(int)] = 1
        
        # Add small epsilon to avoid log(0)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        
        # Calculate cross-entropy
        loss = -np.sum(y_true_one_hot * np.log(y_pred)) / m
        
        return loss
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Backward pass through the network to update weights and biases.
        
        Args:
            X: Input data of shape (n_samples, input_size).
            y: True labels (indices) of shape (n_samples,).
        """
        m = X.shape[0]
        
        # Convert y to one-hot encoding
        y_one_hot = np.zeros((m, self.layer_sizes[-1]))
        y_one_hot[np.arange(m), y.astype(int)] = 1
        
        # Calculate initial error (delta) for output layer
        # For softmax + cross-entropy, this simplifies to (A_L - y)
        delta = self.A_values[-1] - y_one_hot
        
        # Backpropagate the error
        for l in range(self.num_layers - 2, -1, -1):
            # Calculate gradients for weights and biases
            dW = np.dot(self.A_values[l].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights and biases
            self.weights[l] -= self.learning_rate * dW
            self.biases[l] -= self.learning_rate * db
            
            # Calculate delta for previous layer (if not the input layer)
            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self.relu_derivative(self.Z_values[l-1])
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, epochs: int = 100) -> Dict[str, List[float]]:
        """
        Train the neural network.
        
        Args:
            X: Training data of shape (n_samples, input_size).
            y: Training labels of shape (n_samples,).
            X_val: Validation data (optional).
            y_val: Validation labels (optional).
            epochs: Number of training epochs.
            
        Returns:
            Dictionary containing training and validation loss history.
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Forward and backward passes
            y_pred = self.forward(X)
            self.backward(X, y)
            
            # Calculate training metrics
            train_loss = self.cross_entropy_loss(y, y_pred)
            train_accuracy = self.accuracy(y, y_pred)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            
            # Calculate validation metrics if data is provided
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.cross_entropy_loss(y_val, val_pred)
                val_accuracy = self.accuracy(y_val, val_pred)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make class predictions for input data.
        
        Args:
            X: Input data of shape (n_samples, input_size).
            
        Returns:
            Predicted class indices of shape (n_samples,).
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
    
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray = None) -> float:
        """
        Calculate accuracy.
        
        Args:
            y_true: True labels (indices).
            y_pred: Predicted probabilities. If None, will run forward pass.
            
        Returns:
            Accuracy value.
        """
        if y_pred is None:
            y_pred = self.forward(X)
        
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y_true)
    
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray = None) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels (indices).
            y_pred: Predicted probabilities. If None, will run forward pass.
            
        Returns:
            Confusion matrix.
        """
        if y_pred is None:
            y_pred = self.predict(X)
        else:
            y_pred = np.argmax(y_pred, axis=1)
        
        n_classes = self.layer_sizes[-1]
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for i in range(len(y_true)):
            conf_matrix[y_true[i], y_pred[i]] += 1
            
        return conf_matrix