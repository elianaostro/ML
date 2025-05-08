# src/pytorch_network.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple

class PyTorchNetwork(nn.Module):
    """
    Neural network implementation using PyTorch.
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001,
                 l2_lambda: float = 0.0, dropout_rate: float = 0.0):
        """
        Initialize a PyTorch neural network with specified layer sizes.
        
        Args:
            layer_sizes: List containing the number of neurons in each layer.
                         First element is input size, last element is output size.
            learning_rate: Learning rate for optimizer.
            l2_lambda: L2 regularization strength (weight decay).
            dropout_rate: Dropout probability.
        """
        super(PyTorchNetwork, self).__init__()
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        
        # Build the network layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            # Add linear layer
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # Add activation and dropout for all but the last layer
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=l2_lambda
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                   epochs: int = 50, batch_size: int = 64,
                   early_stopping_patience: Optional[int] = None,
                   verbose: int = 1) -> Dict[str, Any]:
        """
        Train the PyTorch neural network.
        
        Args:
            X_train: Training data.
            y_train: Training labels.
            X_val: Validation data.
            y_val: Validation labels.
            epochs: Number of training epochs.
            batch_size: Size of mini-batches.
            early_stopping_patience: Number of epochs to wait for improvement.
            verbose: Verbosity level (0: silent, 1: progress bar).
            
        Returns:
            Dictionary containing training history.
        """
        from src.utils import update_progress_bar
        
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Initialize history
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
        
        # For early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for inputs, targets in train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()
            
            # Calculate average loss and accuracy for the epoch
            train_loss = train_loss / total_train
            train_accuracy = correct_train / total_train
            
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            
            # Validation phase
            val_loss = None
            val_accuracy = None
            
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(outputs, y_val_tensor).item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_accuracy = (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
                
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state_dict = self.model.state_dict().copy()
                    history['best_epoch'] = epoch + 1
                    history['best_val_loss'] = val_loss
                    history['best_val_accuracy'] = val_accuracy
                else:
                    patience_counter += 1
            
            # Update progress bar
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
            
            # Early stopping check
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                if verbose >= 1:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                
                # Restore best weights
                if best_state_dict is not None:
                    self.model.load_state_dict(best_state_dict)
                break
        
        # Print newline after progress bar
        if verbose >= 1:
            print()
        
        # Record total training time
        history['training_time'] = time.time() - start_time
        
        # Print final results
        if verbose >= 1:
            print(f"\nTraining completed in {history['training_time']:.2f} seconds")
            print(f"Final train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.4f}")
            if val_loss is not None:
                print(f"Final val loss: {val_loss:.4f}, val accuracy: {val_accuracy:.4f}")
                print(f"Best epoch: {history['best_epoch']}")
                print(f"Best val loss: {history['best_val_loss']:.4f}, best val accuracy: {history['best_val_accuracy']:.4f}")
        
        return history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Evaluate the model on new data.
        
        Args:
            X: Input data.
            y: True labels.
            
        Returns:
            Tuple containing accuracy, loss, and predictions.
        """
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Compute predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor).item()
            _, predicted = torch.max(outputs.data, 1)
        
        # Calculate accuracy
        accuracy = (predicted == y_tensor).sum().item() / y_tensor.size(0)
        
        return accuracy, loss, predicted.numpy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data.
            
        Returns:
            Predicted classes.
        """
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Compute predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions for each class.
        
        Args:
            X: Input data.
            
        Returns:
            Class probabilities.
        """
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Compute probabilities
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
        
        return probabilities.numpy()
    
    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the confusion matrix.
        
        Args:
            X: Input data.
            y: True labels.
            
        Returns:
            Confusion matrix.
        """
        # Get predictions
        y_pred = self.predict(X)
        
        # Create confusion matrix
        n_classes = self.layer_sizes[-1]
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for i in range(len(y)):
            conf_matrix[y[i], y_pred[i]] += 1
            
        return conf_matrix