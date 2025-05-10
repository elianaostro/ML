import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import time

def run_experiments(X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray,
                    neural_network_class, layer_sizes: List[int],
                    experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run experiments with different neural network configurations.
    """
    results = []
    
    for exp in experiments:
        print(f"\nRunning experiment: {exp['name']}")
        
        model_params = {k: v for k, v in exp.items() if k not in ['name', 'epochs', 'batch_size', 'optimizer', 
                                                                 'lr_schedule', 'early_stopping_patience']}
        model = neural_network_class(layer_sizes=layer_sizes, **model_params)
        
        train_params = {k: v for k, v in exp.items() if k in ['epochs', 'batch_size', 'optimizer', 
                                                             'lr_schedule', 'early_stopping_patience']}
        
        history = model.train(X_train, y_train, X_val, y_val, verbose=1, **train_params)
        
        exp_result = exp.copy()
        exp_result.update({
            'model': model,
            'history': history,
            'final_train_loss': history['train_loss'][-1],
            'final_train_accuracy': history['train_accuracy'][-1],
            'final_val_loss': history['val_loss'][-1] if 'val_loss' in history and history['val_loss'] else None,
            'final_val_accuracy': history['val_accuracy'][-1] if 'val_accuracy' in history and history['val_accuracy'] else None,
            'training_time': history['training_time']
        })

        print(f"\nResults for {exp['name']}:")
        print(f"Training time: {history['training_time']:.2f} seconds")
        print(f"Final train loss: {exp_result['final_train_loss']:.4f}, train accuracy: {exp_result['final_train_accuracy']:.4f}")
        print(f"Final val loss: {exp_result['final_val_loss']:.4f}, val accuracy: {exp_result['final_val_accuracy']:.4f}")

        results.append(exp_result)
    
    return results

def plot_experiment_results(results: List[Dict[str, Any]], metric: str = 'val_accuracy') -> None:
    """
    Plot experiment results for comparison.
    
    Args:
        results: List of experiment results.
        metric: Metric to compare ('val_accuracy', 'val_loss', 'train_accuracy', 'train_loss', 'accuracy_val_vs_train').
    """
    plt.figure(figsize=(12, 8))
    
    for res in results:
        if metric in res['history']:
            plt.plot(res['history'][metric], label=res['name'])
        elif metric == 'accuracy_val_vs_train':
            plt.plot(res['history']['val_accuracy'], label=f"{res['name']} - Val Accuracy")
            plt.plot(res['history']['train_accuracy'], label=f"{res['name']} - Train Accuracy")
    
    plt.title(f"Comparison of {metric.replace('_', ' ').title()}")
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def run_architecture_experiments(architectures, cofiguration, ImprovedNeuralNetwork, 
                                 X_train, y_train, X_val, y_val):
    """
    Run experiments with different neural network architectures.

    Parameters:
    - architectures (list): List of dicts, each with 'name' and 'layer_sizes' keys.
    - cofiguration (dict): configuration dictionary.
    - ImprovedNeuralNetwork (class): Class of the neural network to instantiate.
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.

    Returns:
    - architecture_results (list): List of dictionaries with results for each architecture.
    """
    architecture_results = []

    for arch in architectures:
        print(f"\n{'='*50}")
        print(f"Testing architecture: {arch['name']}")
        print(f"{'='*50}")

        config = {k: v for k, v in cofiguration.items() if k not in ['name', 'model', 'history', 'final_train_loss', 
                                                                    'final_train_accuracy', 'final_val_loss', 
                                                                    'final_val_accuracy', 'training_time']}
        config['name'] = arch['name']

        model = ImprovedNeuralNetwork(layer_sizes=arch['layer_sizes'], **{k: v for k, v in config.items() 
                                                                          if k not in ['name', 'epochs', 'batch_size', 
                                                                                     'optimizer', 'lr_schedule', 
                                                                                     'early_stopping_patience']})

        train_params = {k: v for k, v in config.items() if k in ['epochs', 'batch_size', 'optimizer', 
                                                                  'lr_schedule', 'early_stopping_patience']}

        history = model.train(X_train, y_train, X_val, y_val, verbose=1, **train_params)

        arch_result = arch.copy()
        arch_result.update({
            'model': model,
            'history': history,
            'final_train_loss': history['train_loss'][-1],
            'final_train_accuracy': history['train_accuracy'][-1],
            'final_val_loss': history['val_loss'][-1] if 'val_loss' in history and history['val_loss'] else None,
            'final_val_accuracy': history['val_accuracy'][-1] if 'val_accuracy' in history and history['val_accuracy'] else None,
            'training_time': history['training_time']
        })

        print(f"\nResults for {arch['name']}:")
        print(f"Training time: {history['training_time']:.2f} seconds")
        print(f"Final train loss: {history['final_train_loss']:.4f}, train accuracy: {history['final_train_accuracy']:.4f}")
        print(f"Final val loss: {history['final_val_loss']:.4f}, val accuracy: {history['final_val_accuracy']:.4f}")

        architecture_results.append(arch_result)
    
    return architecture_results

def compare_training_times(results: List[Dict[str, Any]]) -> None:
    """
    Plot training times for different experiments.
    
    Args:
        results: List of experiment results.
    """
    names = [res['name'] for res in results]
    times = [res['training_time'] for res in results]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, times)
    
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{time_val:.2f}s", ha='center')
    
    plt.title('Training Time Comparison')
    plt.xlabel('Experiment')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def compare_final_metrics(results: List[Dict[str, Any]]) -> None:
    """
    Plot final metrics for different experiments.
    
    Args:
        results: List of experiment results.
    """
    names = [res['name'] for res in results]
    train_acc = [res['final_train_accuracy'] for res in results]
    val_acc = [res['final_val_accuracy'] for res in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, train_acc, width, label='Train Accuracy')
    bars2 = plt.bar(x + width/2, val_acc, width, label='Validation Accuracy')
    
    for bar, val in zip(bars1, train_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', fontsize=8)
    
    for bar, val in zip(bars2, val_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', fontsize=8)
    
    plt.title('Final Accuracy Comparison')
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()