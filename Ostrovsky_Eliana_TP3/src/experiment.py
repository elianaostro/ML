import numpy as np
from typing import Dict, List, Any

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
            'layer_sizes': layer_sizes,
            'history': history,
            'final_train_loss': history['train_loss'][-1],
            'final_train_accuracy': history['train_accuracy'][-1],
            'final_val_loss': history['val_loss'][-1] if 'val_loss' in history and history['val_loss'] else None,
            'final_val_accuracy': history['val_accuracy'][-1] if 'val_accuracy' in history and history['val_accuracy'] else None,
            'training_time': history['training_time']
        })

        results.append(exp_result)
    
    return results

def run_architecture_experiments(architectures, configuration, NeuralNetwork, 
                                 X_train, y_train, X_val, y_val):
    """
    Run experiments with different neural network architectures.

    Parameters:
    - architectures (list): List of dicts, each with 'name' and 'layer_sizes' keys.
    - configuration (dict): configuration dictionary.
    - NeuralNetwork (class): Class of the neural network to instantiate.
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.

    Returns:
    - architecture_results (list): List of dictionaries with results for each architecture.
    """
    architecture_results = []

    for arch in architectures:
        print(f"Testing architecture: {arch['name']}")

        config = {k: v for k, v in configuration.items() if k not in ['name', 'model', 'history', 'final_train_loss', 
                                                                    'final_train_accuracy', 'final_val_loss', 
                                                                    'final_val_accuracy', 'training_time']}
        config['name'] = arch['name']

        model = NeuralNetwork(layer_sizes=arch['layer_sizes'], **{k: v for k, v in config.items() 
                                                                          if k not in ['name', 'epochs', 'batch_size', 
                                                                                     'optimizer', 'lr_schedule', 
                                                                                     'early_stopping_patience', 'layer_sizes']})

        train_params = {k: v for k, v in config.items() if k in ['epochs', 'batch_size', 'optimizer', 
                                                                  'lr_schedule', 'early_stopping_patience']}

        history = model.train(X_train, y_train, X_val, y_val, verbose=1, **train_params)

        arch_result = arch.copy()
        arch_result.update(config)
        arch_result.update({
            'model': model,
            'history': history,
            'final_train_loss': history['train_loss'][-1],
            'final_train_accuracy': history['train_accuracy'][-1],
            'final_val_loss': history['val_loss'][-1] if 'val_loss' in history and history['val_loss'] else None,
            'final_val_accuracy': history['val_accuracy'][-1] if 'val_accuracy' in history and history['val_accuracy'] else None,
            'training_time': history['training_time']
        })

        architecture_results.append(arch_result)
    
    return architecture_results
