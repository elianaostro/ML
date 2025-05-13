import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple

class PyTorchNetwork(nn.Module):
    """
    Neural network implementation using PyTorch.
    All inputs and outputs use numpy.ndarray.
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001,
                 l2_lambda: float = 0.0, dropout_rate: float = 0.0):
        """
        Initialize the PyTorchNetwork.

        Args:
            layer_sizes (List[int]): Sizes of each layer, including input and output.
            learning_rate (float): Learning rate for the optimizer.
            l2_lambda (float): L2 regularization parameter.
            dropout_rate (float): Dropout rate applied to hidden layers.
        """
        super(PyTorchNetwork, self).__init__()
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

        self.model = nn.Sequential(*layers)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=l2_lambda
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output probabilities after applying softmax.
        """
        x_tensor = torch.from_numpy(x.astype(np.float32))
        output = self.model(x_tensor)
        probabilities = nn.functional.softmax(output, dim=1)
        return probabilities.detach().numpy()

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                    epochs: int = 50, batch_size: int = 64,
                    early_stopping_patience: Optional[int] = None,
                    verbose: int = 1) -> Dict[str, Any]:
        """
        Train the neural network.

        Args:
            X_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training labels.
            X_val (Optional[np.ndarray]): Validation input data.
            y_val (Optional[np.ndarray]): Validation labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            early_stopping_patience (Optional[int]): Patience for early stopping.
            verbose (int): Verbosity level.

        Returns:
            Dict[str, Any]: Training history including losses and accuracies.
        """
        from src.utils import update_progress_bar

        X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
        y_train_tensor = torch.from_numpy(y_train.astype(np.int64))

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.from_numpy(X_val.astype(np.float32))
            y_val_tensor = torch.from_numpy(y_val.astype(np.int64))

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

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
        best_state_dict = None

        start_time = time.time()

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()

            train_loss = train_loss / total_train
            train_accuracy = correct_train / total_train

            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)

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

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state_dict = self.model.state_dict().copy()
                    history['best_epoch'] = epoch + 1
                    history['best_val_loss'] = val_loss
                    history['best_val_accuracy'] = val_accuracy
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
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                if best_state_dict is not None:
                    self.model.load_state_dict(best_state_dict)
                break

        if verbose >= 1:
            print()

        history['training_time'] = time.time() - start_time

        if verbose >= 2:
            print(f"\nTraining completed in {history['training_time']:.2f} seconds")
            print(f"Final train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.4f}")
            if val_loss is not None:
                print(f"Final val loss: {val_loss:.4f}, val accuracy: {val_accuracy:.4f}")
                print(f"Best epoch: {history['best_epoch']}")
                print(f"Best val loss: {history['best_val_loss']:.4f}, best val accuracy: {history['best_val_accuracy']:.4f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted class labels.
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
    
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy of predictions.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy.
        """
        return np.mean(y_true == y_pred)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Evaluate the model on new data.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.

        Returns:
            Tuple[float, float, np.ndarray]: Accuracy, loss, and predictions.
        """
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.int64))

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor).item()
            _, predicted = torch.max(outputs.data, 1)

        accuracy = self.accuracy(y, predicted.numpy())
        
        return accuracy, loss, predicted.numpy()

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the confusion matrix.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.

        Returns:
            np.ndarray: Confusion matrix.
        """
        y_pred = self.predict(X)
        n_classes = self.layer_sizes[-1]
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

        for i in range(len(y)):
            conf_matrix[y[i], y_pred[i]] += 1

        return conf_matrix
