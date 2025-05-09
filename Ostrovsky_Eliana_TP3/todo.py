import numpy as np
import matplotlib.pyplot as plt
import sys
from src.preprocessing import *
from src.plots import *
from src.visualization import *
from src.neural_network import *
from src.improved_neural_network import *
from src.experiment import *
from src.utils import *

X_images = np.load("Ostrovsky_Eliana_TP3/data/X_images.npy")
y_images = np.load("Ostrovsky_Eliana_TP3/data/y_images.npy")

X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
    X_images / 255.0 , y_images, train_ratio=0.8 * 0.8, val_ratio=0.8 * 0.2, test_ratio=0.2, random_state=42
)

# Setup configuration for experiments
input_size = X_train.shape[1]  # 784 for 28x28 images
output_size = len(np.unique(y_train))  # Number of classes (49)
layer_sizes = [input_size, 100, 80, output_size]  # Same architecture as M0

"""
# Define experiments to run
experiments = [
    # Baseline model (SGD without any improvements)
    {
        'name': 'Baseline (SGD)',
        'learning_rate': 0.01,
        'l2_lambda': 0.0,
        'dropout_rate': 0.0,
        'use_batch_norm': False,
        'epochs': 2000,
        'batch_size': None,  # Full batch
        'optimizer': 'sgd',
        'lr_schedule': None,
        'early_stopping_patience': None
    },
    
    # Rate scheduling (Linear)
    {
        'name': 'Linear Rate Scheduling',
        'learning_rate': 0.01,
        'l2_lambda': 0.0,
        'dropout_rate': 0.0,
        'use_batch_norm': False,
        'epochs': 2000,
        'batch_size': None,
        'optimizer': 'sgd',
        'lr_schedule': 'linear',
        'early_stopping_patience': None
    },
    
    # Rate scheduling (Exponential)
    {
        'name': 'Exponential Rate Scheduling',
        'learning_rate': 0.01,
        'l2_lambda': 0.0,
        'dropout_rate': 0.0,
        'use_batch_norm': False,
        'epochs': 2000,
        'batch_size': None,
        'optimizer': 'sgd',
        'lr_schedule': 'exponential',
        'early_stopping_patience': None
    },
    
    # Mini-batch SGD
    {
        'name': 'Mini-batch SGD',
        'learning_rate': 0.001,
        'l2_lambda': 0.0,
        'dropout_rate': 0.0,
        'use_batch_norm': False,
        'epochs': 2000,
        'batch_size': 64,  # Mini-batch size
        'optimizer': 'sgd',
        'lr_schedule': None,
        'early_stopping_patience': None
    },
    
    # Adam optimizer
    {
        'name': 'ADAM Optimizer',
        'learning_rate': 0.001,
        'l2_lambda': 0.0,
        'dropout_rate': 0.0,
        'use_batch_norm': False,
        'epochs': 2000,
        'batch_size': 64,  # Mini-batch size (Adam works better with mini-batches)
        'optimizer': 'adam',
        'lr_schedule': None,
        'early_stopping_patience': None
    },
    
    # L2 Regularization
    {
        'name': 'L2 Regularization',
        'learning_rate': 0.001,
        'l2_lambda': 0.001,  # L2 regularization strength
        'dropout_rate': 0.0,
        'use_batch_norm': False,
        'epochs': 2000,
        'batch_size': 64,
        'optimizer': 'sgd',
        'lr_schedule': None,
        'early_stopping_patience': None
    },
    
    # Early Stopping
    {
        'name': 'Early Stopping',
        'learning_rate': 0.001,
        'l2_lambda': 0.0,
        'dropout_rate': 0.0,
        'use_batch_norm': False,
        'epochs': 4000,  # More epochs, but we'll stop early
        'batch_size': 64,
        'optimizer': 'sgd',
        'lr_schedule': None,
        'early_stopping_patience': 10  # Stop if no improvement for 10 epochs
    },
    
    # Dropout (optional)
    {
        'name': 'Dropout',
        'learning_rate': 0.001,
        'l2_lambda': 0.0,
        'dropout_rate': 0.2,  # 20% dropout rate
        'use_batch_norm': False,
        'epochs': 2000,
        'batch_size': 64,
        'optimizer': 'sgd',
        'lr_schedule': None,
        'early_stopping_patience': None
    },
    
    # Batch Normalization (optional)
    {
        'name': 'Batch Normalization',
        'learning_rate': 0.001,
        'l2_lambda': 0.0,
        'dropout_rate': 0.0,
        'use_batch_norm': True,  # Enable batch normalization
        'epochs': 2000,
        'batch_size': 64,
        'optimizer': 'sgd',
        'lr_schedule': None,
        'early_stopping_patience': None
    },
    
    # Combined improvements (find optimal configuration)
    {
        'name': 'Combined Improvements',
        'learning_rate': 0.001,
        'l2_lambda': 0.0005,  # Moderate L2 regularization
        'dropout_rate': 0.2,   # Moderate dropout
        'use_batch_norm': True,  # Use batch normalization
        'epochs': 4000,
        'batch_size': 64,
        'optimizer': 'adam',   # Adam optimizer
        'lr_schedule': 'exponential',  # Learning rate scheduling
        'early_stopping_patience': 15  # Early stopping
    }
]

# Run all experiments
results = run_experiments(X_train, y_train, X_val, y_val, ImprovedNeuralNetwork, layer_sizes, experiments)

# Plot learning curves for all experiments
plot_experiment_results(results, metric='val_accuracy')
plot_experiment_results(results, metric='val_loss')

# Find the best model based on validation accuracy
best_result = max(results, key=lambda x: x['final_val_accuracy'])
print(f"\nBest model: {best_result['name']}")
print(f"Validation accuracy: {best_result['final_val_accuracy']:.4f}")

# Let's explore different architectures with the best configuration
print("\nExploring different architectures with the best configuration...")
""" 
architectures = [
    {
        'name': 'Single Hidden Layer (200)',
        'layer_sizes': [input_size, 200, output_size]
    },
    {
        'name': 'Two Hidden Layers (100, 80)',
        'layer_sizes': [input_size, 100, 80, output_size]
    },
    {
        'name': 'Three Hidden Layers (120, 80, 60)',
        'layer_sizes': [input_size, 120, 80, 60, output_size]
    },
    {
        'name': 'Wide Network (200, 150)',
        'layer_sizes': [input_size, 200, 150, output_size]
    },
    {
        'name': 'Deep Network (100, 80, 60, 40)',
        'layer_sizes': [input_size, 100, 80, 60, 40, output_size]
    }
]

architecture_results = run_architecture_experiments(
    architectures=architectures,
    cofiguration=best_result,
    ImprovedNeuralNetwork=ImprovedNeuralNetwork,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val
)

compare_final_metrics(architecture_results)

best_arch_result = max(architecture_results, key=lambda x: x['final_val_accuracy'])

# src/todo.py (continuación - agregar al final)

# Asegurarse de que las variables necesarias como X_train, y_train, X_val, y_val,
# la clase ImprovedNeuralNetwork, y las funciones run_experiments, plot_experiment_results,
# compare_final_metrics estén definidas y accesibles en este punto del script.

# También, se asume que la variable 'best_arch_result' contiene los resultados
# del mejor modelo encontrado en la fase de exploración de arquitecturas,
# y 'input_size', 'output_size' están definidas.

print("\n######################################################################")
print("## Ejecutando Barrido de Tasa de Aprendizaje para la Mejor Arquitectura ##")
print("######################################################################")

# Definir un rango de tasas de aprendizaje para probar
# Puedes ajustar esta lista según sea necesario
learning_rates_to_test = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

lr_tuning_experiments = []

# 'best_arch_result' debería estar disponible de la parte anterior de tu script.
# Contiene la configuración y los resultados del mejor modelo de la búsqueda de arquitecturas.

# Hiperparámetros a copiar de best_arch_result
# Estos son parámetros esperados por ImprovedNeuralNetwork y su método train.
keys_to_copy_from_best_arch = [
'l2_lambda', 'dropout_rate', 'use_batch_norm',
'epochs', 'batch_size', 'optimizer',
'lr_schedule', 'early_stopping_patience'
# 'learning_rate' se copiará inicialmente pero luego se sobrescribirá por cada valor en learning_rates_to_test
]

for lr_value in learning_rates_to_test:
    new_exp_config = {}

    # Copiar hiperparámetros relevantes de best_arch_result
    for key in keys_to_copy_from_best_arch:
        if key in best_arch_result:
            new_exp_config[key] = best_arch_result[key]
        # else:
            # Opcionalmente, manejar claves faltantes.
            # Por ejemplo, podrías establecer un valor predeterminado:
            # if key == 'epochs': new_exp_config[key] = 2000 # Epochs por defecto

# Sobrescribir/establecer la tasa de aprendizaje para este experimento específico
new_exp_config['learning_rate'] = lr_value

# Crear un nombre descriptivo para el experimento
new_exp_config['name'] = f"{best_arch_result['name']}_LR_{lr_value}"

lr_tuning_experiments.append(new_exp_config)

# Ejecutar los experimentos de ajuste de la tasa de aprendizaje
if lr_tuning_experiments:
    print(f"\nIniciando {len(lr_tuning_experiments)} experimentos de ajuste de LR...")

# Asegúrate de que X_train, y_train, X_val, y_val, ImprovedNeuralNetwork,
# run_experiments, plot_experiment_results, compare_final_metrics
# estén disponibles en el alcance (scope) actual.

lr_tuning_results = run_experiments(
    X_train, y_train, X_val, y_val,
    ImprovedNeuralNetwork, # Clase de la red neuronal
    best_arch_result[''], # Arquitectura fija del mejor modelo
    lr_tuning_experiments   # Lista de configuraciones de experimentos (variando LR)
)

# Graficar los resultados del ajuste de la tasa de aprendizaje
if lr_tuning_results:
    print("\nGraficando resultados del ajuste de LR...")
    plot_experiment_results(lr_tuning_results, metric='val_accuracy', title_suffix=" - Ajuste de LR")
    plot_experiment_results(lr_tuning_results, metric='val_loss', title_suffix=" - Ajuste de LR")
    compare_final_metrics(lr_tuning_results, title_suffix=" - Ajuste de LR")

    # Encontrar e imprimir el mejor modelo de la fase de ajuste de LR
    best_lr_tuned_model = max(lr_tuning_results, key=lambda x: x.get('final_val_accuracy', -float('inf')))
    print(f"\n--- Mejor Modelo Después del Ajuste de Tasa de Aprendizaje ---")
    print(f"Nombre: {best_lr_tuned_model.get('name', 'N/A')}")
    print(f"Precisión de Validación: {best_lr_tuned_model.get('final_val_accuracy', 'N/A'):.4f}")
    print(f"Tasa de Aprendizaje: {best_lr_tuned_model.get('learning_rate', 'N/A')}")
    print(f"Configuración Completa:")
    for key, value in best_lr_tuned_model.items():
        if key not in ['history']: # Evitar imprimir la lista larga del historial
                print(f"  {key}: {value}")
            
print("\n--- Fin del Script ---")
