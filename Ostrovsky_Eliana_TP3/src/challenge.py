# src/challenge.py
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional
import os

def generate_predictions(model: Any, X_comp: np.ndarray, output_size: int, 
                        apellido: str, nombre: str) -> str:
    """
    Generate predictions for the challenge dataset and save to CSV.
    
    Args:
        model: Trained model to use for predictions
        X_comp: Competition data
        output_size: Number of classes
        apellido: Last name for filename
        nombre: First name for filename
        
    Returns:
        Path to the generated predictions file
    """
    # Normalize data if not already normalized
    if X_comp.max() > 1.0:
        X_comp = X_comp / 255.0
    
    print(f"Generating predictions for X_COMP shape: {X_comp.shape}")
    
    # Generate probabilities based on model type
    if hasattr(model, 'predict_proba'):
        # For our custom implementation
        predictions_proba = model.predict_proba(X_comp)
    elif isinstance(model, torch.nn.Module):
        # For PyTorch model
        X_comp_tensor = torch.FloatTensor(X_comp)
        model.eval()
        with torch.no_grad():
            outputs = model(X_comp_tensor)
            predictions_proba = torch.nn.functional.softmax(outputs, dim=1).numpy()
    else:
        raise TypeError("Model type not supported for prediction")
    
    # Create dataframe with probabilities
    columns = [f'Clase_{i}' for i in range(output_size)]
    predictions_df = pd.DataFrame(predictions_proba, columns=columns)
    
    # Create filename
    filename = f"{apellido}_{nombre}_predicciones.csv"
    
    # Save to CSV
    predictions_df.to_csv(filename, index=False)
    
    print(f"Predictions successfully saved to: {filename}")
    print(f"Sample of predictions (first row):")
    print(predictions_df.iloc[0])
    
    return filename

# Main execution for notebook
def run_challenge(best_model, X_train, y_train, apellido="Apellido", nombre="Nombre"):
    """
    Run the challenge prediction process.
    
    Args:
        best_model: The best model identified from previous exercises
        X_train: Training data (for reference only)
        y_train: Training labels (for number of classes)
        apellido: Last name for the filename
        nombre: First name for the filename
    """
    print("=" * 50)
    print("EJERCICIO 5: DESAFÍO")
    print("=" * 50)
    
    # Load competition data
    try:
        X_COMP = np.load("X_COMP.npy")
        print(f"Competition data loaded: {X_COMP.shape}")
    except FileNotFoundError:
        print("Error: X_COMP.npy file not found. Make sure it's in the current directory.")
        return
    
    # Get number of classes
    output_size = len(np.unique(y_train))
    print(f"Number of classes: {output_size}")
    
    # Generate and save predictions
    filename = generate_predictions(best_model, X_COMP, output_size, apellido, nombre)
    
    print("\nDesafío completado!")
    print(f"Archivo de predicciones: {filename}")
    
    # Verify the file was created
    if os.path.exists(filename):
        file_size = os.path.getsize(filename) / 1024  # KB
        print(f"Tamaño del archivo: {file_size:.2f} KB")
        
        # Check header and first rows
        with open(filename, 'r') as f:
            header = f.readline().strip()
            first_row = f.readline().strip() if f.readline() else ""
        
        print(f"Encabezado: {header}")
        if first_row:
            print(f"Primera fila de datos: {first_row[:50]}..." if len(first_row) > 50 else first_row)
    else:
        print("¡Advertencia! El archivo no se creó correctamente.")