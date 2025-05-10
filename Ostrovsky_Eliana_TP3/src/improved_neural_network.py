import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional

class ImprovedNeuralNetwork:
    """
    Advanced neural network implementation with various improvements:
    - Learning rate scheduling (linear and exponential)
    - Mini-batch gradient descent
    - ADAM optimizer
    - L2 regularization
    - Early stopping
    - Optional dropout regularization
    - Optional batch normalization
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001,
                 l2_lambda: float = 0.0, dropout_rate: float = 0.0,
                 use_batch_norm: bool = False):
        """
        Initialize an improved neural network with specified parameters.
        
        Args:
            layer_sizes: List containing the number of neurons in each layer.
                         First element is input size, last element is output size.
            learning_rate: Initial learning rate for gradient descent.
            l2_lambda: L2 regularization strength. 0.0 means no regularization.
            dropout_rate: Dropout probability. 0.0 means no dropout.
            use_batch_norm: Whether to use batch normalization.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        self.weights = []
        self.biases = []
        
        self.gamma = [] 
        self.beta = []  
        self.running_mean = []
        self.running_var = [] 
        
        for i in range(1, self.num_layers):
            scale = np.sqrt(2.0 / layer_sizes[i-1])
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * scale)
            self.biases.append(np.zeros((1, layer_sizes[i])))
            
            if use_batch_norm and i < self.num_layers - 1: 
                self.gamma.append(np.ones((1, layer_sizes[i])))
                self.beta.append(np.zeros((1, layer_sizes[i])))
                self.running_mean.append(np.zeros((1, layer_sizes[i])))
                self.running_var.append(np.ones((1, layer_sizes[i])))
        
        self.adam_m = [np.zeros_like(w) for w in self.weights] 
        self.adam_v = [np.zeros_like(w) for w in self.weights] 
        self.adam_m_bias = [np.zeros_like(b) for b in self.biases] 
        self.adam_v_bias = [np.zeros_like(b) for b in self.biases] 
        self.beta1 = 0.9  
        self.beta2 = 0.999
        self.epsilon = 1e-8  
        self.t = 0 
    
    def relu(self, Z: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU activation function."""
        return np.where(Z > 0, 1, 0)
    
    def softmax(self, Z: np.ndarray) -> np.ndarray:
        """Softmax activation function with numerical stability."""
        shift_Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(shift_Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def batch_norm_forward(self, Z: np.ndarray, layer_idx: int, training: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Apply batch normalization to input Z.
        
        Args:
            Z: Input activations.
            layer_idx: Index of the layer.
            training: Whether in training mode.
            
        Returns:
            Normalized activations and cache for backpropagation.
        """
        cache = {}
        
        if training:
            mu = np.mean(Z, axis=0, keepdims=True)
            var = np.var(Z, axis=0, keepdims=True) + self.epsilon
            
            Z_norm = (Z - mu) / np.sqrt(var)
            
            out = self.gamma[layer_idx] * Z_norm + self.beta[layer_idx]
            
            momentum = 0.9
            self.running_mean[layer_idx] = momentum * self.running_mean[layer_idx] + (1 - momentum) * mu
            self.running_var[layer_idx] = momentum * self.running_var[layer_idx] + (1 - momentum) * var
            
            cache = {'Z': Z, 'Z_norm': Z_norm, 'mu': mu, 'var': var, 'gamma': self.gamma[layer_idx], 
                    'beta': self.beta[layer_idx], 'layer_idx': layer_idx}
        else:
            Z_norm = (Z - self.running_mean[layer_idx]) / np.sqrt(self.running_var[layer_idx] + self.epsilon)
            out = self.gamma[layer_idx] * Z_norm + self.beta[layer_idx]
        
        return out, cache
    
    def batch_norm_backward(self, dout: np.ndarray, cache: Dict) -> np.ndarray:
        """
        Backward pass for batch normalization.
        
        Args:
            dout: Gradient from the next layer.
            cache: Cache from the forward pass.
            
        Returns:
            Gradient with respect to input Z.
        """
        Z = cache['Z']
        Z_norm = cache['Z_norm']
        mu = cache['mu']
        var = cache['var']
        gamma = cache['gamma']
        layer_idx = cache['layer_idx']
        N = Z.shape[0]
        
        dgamma = np.sum(dout * Z_norm, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)
        
        self.gamma[layer_idx] -= self.learning_rate * dgamma
        self.beta[layer_idx] -= self.learning_rate * dbeta
        
        dZ_norm = dout * gamma
        
        dvar = np.sum(dZ_norm * (Z - mu) * -0.5 * np.power(var + self.epsilon, -1.5), axis=0, keepdims=True)
        dmu = np.sum(dZ_norm * -1.0 / np.sqrt(var + self.epsilon), axis=0, keepdims=True) + \
              dvar * np.mean(-2.0 * (Z - mu), axis=0, keepdims=True)
        
        dZ = dZ_norm / np.sqrt(var + self.epsilon) + \
             dvar * 2.0 * (Z - mu) / N + \
             dmu / N
        
        return dZ
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            X: Input data of shape (n_samples, input_size).
            training: Whether in training mode (affects dropout and batch norm).
            
        Returns:
            Output predictions of shape (n_samples, output_size).
        """
        self.Z_values = []  
        self.A_values = [X] 
        self.dropout_masks = []  
        self.batch_norm_caches = [] 
        
        for i in range(self.num_layers - 2):
            Z = np.dot(self.A_values[-1], self.weights[i]) + self.biases[i]
            
            if self.use_batch_norm:
                Z, bn_cache = self.batch_norm_forward(Z, i, training)
                self.batch_norm_caches.append(bn_cache)
            
            self.Z_values.append(Z)
            A = self.relu(Z)
            
            if training and self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, size=A.shape) / (1 - self.dropout_rate)
                A *= mask
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
            
            self.A_values.append(A)
        
        Z = np.dot(self.A_values[-1], self.weights[-1]) + self.biases[-1]
        self.Z_values.append(Z)
        A = self.softmax(Z)
        self.A_values.append(A)
        
        return self.A_values[-1]
    
    def cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate cross-entropy loss with L2 regularization if enabled.
        
        Args:
            y_true: True labels (indices).
            y_pred: Predicted probabilities.
            
        Returns:
            Cross-entropy loss value.
        """
        m = y_true.shape[0]
        
        y_true_one_hot = np.zeros((m, self.layer_sizes[-1]))
        y_true_one_hot[np.arange(m), y_true.astype(int)] = 1
        
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        cross_entropy = -np.sum(y_true_one_hot * np.log(y_pred)) / m
        
        l2_term = 0
        if self.l2_lambda > 0:
            for w in self.weights:
                l2_term += np.sum(np.square(w))
            l2_term = (self.l2_lambda / (2 * m)) * l2_term
        
        return cross_entropy + l2_term
    
    def backward(self, X: np.ndarray, y: np.ndarray, optimizer: str = 'sgd') -> None:
        """
        Backward pass through the network to update weights and biases.
        
        Args:
            X: Input data of shape (n_samples, input_size).
            y: True labels (indices) of shape (n_samples,).
            optimizer: Optimization algorithm to use ('sgd' or 'adam').
        """
        m = X.shape[0]
        
        y_one_hot = np.zeros((m, self.layer_sizes[-1]))
        y_one_hot[np.arange(m), y.astype(int)] = 1
        
        delta = self.A_values[-1] - y_one_hot
        
        dW = []
        db = []
        
        dW_out = np.dot(self.A_values[-2].T, delta) / m
        db_out = np.sum(delta, axis=0, keepdims=True) / m
        
        if self.l2_lambda > 0:
            dW_out += (self.l2_lambda / m) * self.weights[-1]
        
        dW.insert(0, dW_out)
        db.insert(0, db_out)
        
        for l in range(self.num_layers - 3, -1, -1):
            delta = np.dot(delta, self.weights[l+1].T)
            
            if self.dropout_masks[l] is not None:
                delta *= self.dropout_masks[l]
            
            delta *= self.relu_derivative(self.Z_values[l])
            
            if self.use_batch_norm:
                delta = self.batch_norm_backward(delta, self.batch_norm_caches[l])
            
            dW_l = np.dot(self.A_values[l].T, delta) / m
            db_l = np.sum(delta, axis=0, keepdims=True) / m
            
            if self.l2_lambda > 0:
                dW_l += (self.l2_lambda / m) * self.weights[l]
            
            dW.insert(0, dW_l)
            db.insert(0, db_l)
        
        if optimizer == 'adam':
            self._update_parameters_adam(dW, db)
        else:
            self._update_parameters_sgd(dW, db)
    
    def _update_parameters_sgd(self, dW: List[np.ndarray], db: List[np.ndarray]) -> None:
        """
        Update parameters using standard gradient descent.
        
        Args:
            dW: List of weight gradients.
            db: List of bias gradients.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def _update_parameters_adam(self, dW: List[np.ndarray], db: List[np.ndarray]) -> None:
        """
        Update parameters using ADAM optimizer.
        
        Args:
            dW: List of weight gradients.
            db: List of bias gradients.
        """
        self.t += 1
        
        for i in range(len(self.weights)):
            self.adam_m[i] = self.beta1 * self.adam_m[i] + (1 - self.beta1) * dW[i]
            self.adam_m_bias[i] = self.beta1 * self.adam_m_bias[i] + (1 - self.beta1) * db[i]
            
            self.adam_v[i] = self.beta2 * self.adam_v[i] + (1 - self.beta2) * (dW[i]**2)
            self.adam_v_bias[i] = self.beta2 * self.adam_v_bias[i] + (1 - self.beta2) * (db[i]**2)
            
            m_corrected = self.adam_m[i] / (1 - self.beta1**self.t)
            m_bias_corrected = self.adam_m_bias[i] / (1 - self.beta1**self.t)
            
            v_corrected = self.adam_v[i] / (1 - self.beta2**self.t)
            v_bias_corrected = self.adam_v_bias[i] / (1 - self.beta2**self.t)
            
            self.weights[i] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_bias_corrected / (np.sqrt(v_bias_corrected) + self.epsilon)
    
    def _update_learning_rate(self, epoch: int, schedule: str, total_epochs: int) -> None:
        """
        Update learning rate according to the specified schedule.
        
        Args:
            epoch: Current epoch number.
            schedule: The learning rate schedule ('linear' or 'exponential').
            total_epochs: Total number of epochs.
        """
        if schedule == 'linear':
            decay = epoch / total_epochs
            self.learning_rate = self.initial_learning_rate * (1.0 / (1.0 + decay))
        elif schedule == 'exponential':
            decay_rate = 0.1
            self.learning_rate = self.initial_learning_rate * np.exp(-decay_rate * epoch / total_epochs)
    
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
            y_val: np.ndarray = None, epochs: int = 100, batch_size: int = None, 
            optimizer: str = 'sgd', lr_schedule: str = None, 
            early_stopping_patience: int = None, verbose: int = 1) -> Dict[str, Any]:
        """
        Train the neural network with various improvements.
        
        Args:
            X: Training data of shape (n_samples, input_size).
            y: Training labels of shape (n_samples,).
            X_val: Validation data (optional).
            y_val: Validation labels (optional).
            epochs: Number of training epochs.
            batch_size: Size of mini-batches. If None, use full batch.
            optimizer: Optimization algorithm ('sgd' or 'adam').
            lr_schedule: Learning rate schedule ('linear' or 'exponential' or None).
            early_stopping_patience: Number of epochs to wait for improvement.
            verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch).
            
        Returns:
            Dictionary containing training history and metrics.
        """
        import time
        from src.utils import update_progress_bar
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
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
            if lr_schedule:
                self._update_learning_rate(epoch, lr_schedule, epochs)
                history['learning_rate'].append(self.learning_rate)
            
            if batch_size is None:
                y_pred = self.forward(X, training=True)
                self.backward(X, y, optimizer)
            else:
                indices = np.random.permutation(m)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                total_batches = (m + batch_size - 1) // batch_size
                for i in range(0, m, batch_size):
                    end = min(i + batch_size, m)
                    X_batch = X_shuffled[i:end]
                    y_batch = y_shuffled[i:end]
                    
                    self.forward(X_batch, training=True)
                    self.backward(X_batch, y_batch, optimizer)
                    
                    if verbose == 2:
                        batch_idx = i // batch_size + 1
                        update_progress_bar(batch_idx, total_batches, 
                                        metrics={"epoch": epoch+1, "total_epochs": epochs})
            
            y_pred_train = self.forward(X, training=False)
            train_loss = self.cross_entropy_loss(y, y_pred_train)
            train_accuracy = self.accuracy(y, y_pred_train)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            
            val_loss = None
            val_accuracy = None
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward(X_val, training=False)
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
                    if self.use_batch_norm:
                        best_gamma = [g.copy() for g in self.gamma]
                        best_beta = [b.copy() for b in self.beta]
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
                metrics["lr"] = self.learning_rate
                    
                update_progress_bar(epoch + 1, epochs, metrics=metrics)
            
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                if verbose >= 1:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                
                self.weights = best_weights
                self.biases = best_biases
                if self.use_batch_norm:
                    self.gamma = best_gamma
                    self.beta = best_beta
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
        """Make class predictions for input data."""
        probabilities = self.forward(X, training=False)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for each class."""
        return self.forward(X, training=False)
    
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray = None) -> float:
        """Calculate accuracy."""
        if y_pred is None:
            y_pred = self.forward(X_true, training=False)
        
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            predictions = np.argmax(y_pred, axis=1)
        else:
            predictions = y_pred
        
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
        """Calculate confusion matrix."""
        if y_pred is None:
            y_pred = self.forward(X_true, training=False)
            predictions = np.argmax(y_pred, axis=1)
        elif y_pred.ndim > 1 and y_pred.shape[1] > 1:
            predictions = np.argmax(y_pred, axis=1)
        else:
            predictions = y_pred
        
        n_classes = self.layer_sizes[-1]
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for i in range(len(y_true)):
            conf_matrix[y_true[i], predictions[i]] += 1
            
        return conf_matrix