import numpy as np
from typing import List, Tuple, Dict, Any
import time
from src.utils import update_progress_bar

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
        
        for i in range(1, self.num_layers):
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
        self.Z_values = []  
        self.A_values = [X]  

        for i in range(self.num_layers - 2):
            Z = np.dot(self.A_values[-1], self.weights[i]) + self.biases[i]
            self.Z_values.append(Z)
            A = self.relu(Z)
            self.A_values.append(A)
        
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
        
        y_true_one_hot = np.zeros((m, self.layer_sizes[-1]))
        y_true_one_hot[np.arange(m), y_true.astype(int)] = 1
        
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        
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
        
        y_one_hot = np.zeros((m, self.layer_sizes[-1]))
        y_one_hot[np.arange(m), y.astype(int)] = 1
        
        delta = self.A_values[-1] - y_one_hot
        
        for l in range(self.num_layers - 2, -1, -1):
            dW = np.dot(self.A_values[l].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            self.weights[l] -= self.learning_rate * dW
            self.biases[l] -= self.learning_rate * db
            
            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self.relu_derivative(self.Z_values[l-1])
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
            y_val: np.ndarray = None, epochs: int = 100, batch_size: int = None,
            early_stopping_patience: int = None, verbose: int = 1) -> Dict[str, Any]:
        """
        Train the neural network.
        
        Args:
            X: Training data of shape (n_samples, input_size).
            y: Training labels of shape (n_samples,).
            X_val: Validation data (optional).
            y_val: Validation labels (optional).
            epochs: Number of training epochs.
            batch_size: Size of mini-batches. If None, use full batch.
            early_stopping_patience: Number of epochs to wait for improvement.
            verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch).
            
        Returns:
            Dictionary containing training history and metrics.
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'training_time': 0,
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'best_val_accuracy': 0.0
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        best_epoch = 0
        
        m = X.shape[0]
        start_time = time.time()
        
        for epoch in range(epochs):
            if batch_size is None:
                y_pred = self.forward(X)
                self.backward(X, y)
            else:
                indices = np.random.permutation(m)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                total_batches = (m + batch_size - 1) // batch_size
                for i in range(0, m, batch_size):
                    end = min(i + batch_size, m)
                    X_batch = X_shuffled[i:end]
                    y_batch = y_shuffled[i:end]
                    
                    self.forward(X_batch)
                    self.backward(X_batch, y_batch)
                    
                    if verbose == 2:
                        batch_idx = i // batch_size + 1
                        update_progress_bar(batch_idx, total_batches, 
                                        metrics={"epoch": epoch+1, "total_epochs": epochs})
            
            y_pred_train = self.forward(X)
            train_loss = self.cross_entropy_loss(y, y_pred_train)
            train_accuracy = self.accuracy(y, y_pred_train)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            
            val_loss = None
            val_accuracy = None
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward(X_val)
                val_loss = self.cross_entropy_loss(y_val, y_pred_val)
                val_accuracy = self.accuracy(y_val, y_pred_val)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch + 1
                    history['best_epoch'] = best_epoch
                    history['best_val_loss'] = val_loss
                    history['best_val_accuracy'] = val_accuracy
                    
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
            
            if verbose >= 1:
                metrics = {
                    "train_loss": train_loss, 
                    "train_acc": train_accuracy
                }
                if val_loss is not None:
                    metrics.update({
                        "val_loss": val_loss, 
                        "val_acc": val_accuracy
                    })
                    
                update_progress_bar(epoch + 1, epochs, metrics=metrics)
            
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                if verbose >= 1:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                
                self.weights = best_weights
                self.biases = best_biases
                break
        
        if verbose >= 1:
            print()
        
        history['training_time'] = time.time() - start_time
        
        if verbose >= 2:
            print(f"\nTraining completed in {history['training_time']:.2f} seconds")
            print(f"Best epoch: {history['best_epoch']}")
            print(f"Final train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.4f}")
            if val_loss is not None:
                print(f"Final val loss: {val_loss:.4f}, val accuracy: {val_accuracy:.4f}")
                print(f"Best val loss: {history['best_val_loss']:.4f}, best val accuracy: {history['best_val_accuracy']:.4f}")
        
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
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Evaluate the model on a dataset.
        
        Args:
            X: Input data of shape (n_samples, input_size).
            y: True labels (indices) of shape (n_samples,).
            
        Returns:
            Tuple containing accuracy and confusion matrix.
        """
        probs = self.forward(X)
        preds = np.argmax(probs, axis=1)
        loss = self.cross_entropy_loss(y, probs)
        acc = self.accuracy(y, probs)
        return acc, loss, preds
    
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
        elif y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        n_classes = self.layer_sizes[-1]
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for i in range(len(y_true)):
            conf_matrix[y_true[i], y_pred[i]] += 1
            
        return conf_matrix