import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional
from metrics import *

def create_stratified_k_folds(X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, k: int = 5, random_state: Optional[int] = None
                              ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates indices for K-Fold cross-validation with stratification.

    Ensures that the proportion of samples for each class is approximately 
    the same across all folds as in the original dataset.

    Args:
        X (Union[pd.DataFrame, np.ndarray]): Feature data. Only its length is used, 
            but passed for API consistency.
        y (np.ndarray): Array of target labels.
        k (int, optional): The number of folds. Must be at least 2. Defaults to 5.
        random_state (int, optional): Seed for the random number generator for 
            reproducible fold assignments. Defaults to None.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of length `k`. Each element is a tuple 
            containing two NumPy arrays: (train_indices, validation_indices) for that fold.
            
    Raises:
        ValueError: If k is less than 2 or greater than the number of samples 
                    in the smallest class.
    """
    if k < 2:
        raise ValueError("Number of folds k must be at least 2.")
        
    if random_state is not None:
        np.random.seed(random_state)

    y_arr = np.asarray(y)
    n_samples = len(y_arr)
    indices = np.arange(n_samples)
    
    unique_labels, y_inversed = np.unique(y_arr, return_inverse=True)
    class_counts = np.bincount(y_inversed)
    
    min_class_size = np.min(class_counts)
    if k > min_class_size:
        raise ValueError(f"Cannot create {k} folds with stratification. The smallest "
                         f"class has only {min_class_size} samples. Reduce k or handle "
                         f"small classes.")

    per_fold_indices: List[List[int]] = [[] for _ in range(k)]
    
    for class_label_idx, count in enumerate(class_counts):
        class_indices_original = indices[y_inversed == class_label_idx]
        np.random.shuffle(class_indices_original)
        
        for i, idx in enumerate(class_indices_original):
            target_fold = i % k
            per_fold_indices[target_fold].append(idx)

    fold_splits: List[Tuple[np.ndarray, np.ndarray]] = []
    all_indices_set = set(indices)
    
    for i in range(k):
        val_indices = np.array(per_fold_indices[i], dtype=int)
        
        val_indices_set = set(val_indices)
        train_indices = np.array(list(all_indices_set - val_indices_set), dtype=int)
        
        fold_splits.append((train_indices, val_indices))

    return fold_splits


def grid_search_cv(model_class, param_configs, X_train, y_train, X_val=None, y_val=None, 
                   n_folds=5, random_seed=42, metric_name='f1_score', verbose=True,
                   preprocess_fn=None, preprocess_info=None):
    """
    Realiza una búsqueda de hiperparámetros con validación cruzada para cualquier modelo.
    
    Args:
        model_class: La clase del modelo a evaluar (ej. LogisticRegression, RandomForest)
        param_configs: Lista de diccionarios donde cada uno contiene una configuración de hiperparámetros
        X_train: DataFrame o array con los datos de entrenamiento
        y_train: Array con las etiquetas de entrenamiento
        X_val: DataFrame o array con los datos de validación (opcional)
        y_val: Array con las etiquetas de validación (opcional)
        n_folds: Número de folds para la validación cruzada
        random_seed: Semilla para reproducibilidad
        metric_name: Nombre de la métrica a optimizar
        verbose: Si es True, imprime información detallada durante la búsqueda
        preprocess_fn: Función opcional para preprocesar los datos. Debe tener la siguiente firma:
                       (X_train_fold, X_val_fold) -> (X_train_processed, X_val_processed)
                       Si es None, no se realiza preprocesamiento.
    
    Returns:
        best_params: La mejor configuración de hiperparámetros encontrada
        best_model: El modelo entrenado con los mejores hiperparámetros
        results: Detalles de la evaluación de todas las configuraciones
    """
    if verbose:
        print(f"Iniciando búsqueda de hiperparámetros con {n_folds}-Fold Cross-Validation")
        print(f"Total de configuraciones a probar: {len(param_configs)}")
        print(f"Métrica a optimizar: {metric_name}")
        print("-" * 60)

    # Preparar folds de CV una sola vez
    cv_folds = create_stratified_k_folds(X_train, y_train, k=n_folds, random_state=random_seed)

    # Variables para seguimiento
    best_avg_score = -1
    best_params = None
    results = []

    # Bucle principal de evaluación de configuraciones
    for i, params in enumerate(param_configs):
        if verbose:
            print(f"\nConfig [{i+1}/{len(param_configs)}]: {params}")
        
        # Evaluación en cada fold
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
            # Obtener datos para este fold
            if isinstance(X_train, pd.DataFrame):
                X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train[train_idx]
                X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train[val_idx]
            else:
                X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
                X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
            
            # Aplicar preprocesamiento si se proporcionó una función
            if preprocess_fn is not None:
                X_train_fold, X_val_fold = preprocess_fn(X_train_fold, X_val_fold, preprocess_info)
            
            # Entrenar modelo con la configuración actual
            model = model_class(**params)
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluar en fold de validación
            y_pred_fold = model.predict(X_val_fold)
            
            # Calcular métrica (manejando diferentes tipos)
            if hasattr(model, 'predict_proba') and metric_name in ['auc_roc', 'auc_pr', 'f1_score']:
                try:
                    y_pred_proba_fold = model.predict_proba(X_val_fold)
                    metrics = calculate_metrics(y_val_fold, y_pred_fold, y_pred_proba_fold)
                    score = metrics[metric_name]
                except:
                    # Si no se puede calcular con probabilidades, usar versión básica
                    if metric_name == 'f1_score':
                        score = f1_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0)
                    elif metric_name == 'accuracy':
                        score = accuracy_score(y_val_fold, y_pred_fold)
                    else:
                        # Para otras métricas, usar F1 como fallback
                        score = f1_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0)
            else:
                # Para modelos sin predict_proba o métricas simples
                if metric_name == 'f1_score':
                    score = f1_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0)
                elif metric_name == 'accuracy':
                    score = accuracy_score(y_val_fold, y_pred_fold)
                else:
                    # Para otras métricas, usar F1 como fallback
                    score = f1_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0)
            
            fold_scores.append(score)
        
        # Calcular y almacenar resultados de esta configuración
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        results.append({
            'params': params,
            'avg_score': avg_score,
            'std_score': std_score,
            'fold_scores': fold_scores
        })
        
        if verbose:
            print(f" Score: {avg_score:.4f} ± {std_score:.4f}")
        
        # Actualizar mejor configuración si corresponde
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_params = params
            if verbose:
                print(f" ¡Nueva mejor configuración encontrada!")

    # Ordenar resultados por puntuación promedio (descendente)
    sorted_results = sorted(results, key=lambda x: x['avg_score'], reverse=True)

    if verbose:
        # Mostrar resultados finales
        print("\n" + "="*60)
        print("RESULTADOS DE LA BÚSQUEDA DE HIPERPARÁMETROS")
        print("="*60)
        print(f"Mejor configuración: {best_params}")
        print(f"Mejor score ({metric_name}): {best_avg_score:.4f}")
        
        # Mostrar top 3 configuraciones
        print("\nTop 3 configuraciones:")
        for i, res in enumerate(sorted_results[:3]):
            print(f"{i+1}. Score: {res['avg_score']:.4f} ± {res['std_score']:.4f} | Params: {res['params']}")
        
        # Visualización simple de resultados
        plt.figure(figsize=(10, 6))
        scores = [r['avg_score'] for r in sorted_results]
        plt.plot(range(len(scores)), scores, 'o-')
        plt.xlabel('Configuración (ordenada por rendimiento)')
        plt.ylabel(f'Score ({metric_name})')
        plt.title('Rendimiento de las configuraciones evaluadas')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Preprocesar datos completos si hay función de preprocesamiento
    if preprocess_fn is not None and X_val is not None:
        X_train_processed, X_val_processed = preprocess_fn(X_train, X_val)
    else:
        X_train_processed, X_val_processed = X_train, X_val

    # Entrenar el modelo final con los mejores parámetros
    best_model = model_class(**best_params)
    best_model.fit(X_train_processed, y_train)

    # Evaluar en conjunto de validación separado si se proporciona
    if X_val_processed is not None and y_val is not None:
        y_val_pred = best_model.predict(X_val_processed)
        
        if hasattr(best_model, 'predict_proba'):
            try:
                y_val_proba = best_model.predict_proba(X_val_processed)
                val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
                
                if verbose:
                    print("\nMétricas en el conjunto de validación:")
                    print(f"Accuracy: {val_metrics['accuracy']:.4f}")
                    print(f"Precision: {val_metrics['precision']:.4f}")
                    print(f"Recall: {val_metrics['recall']:.4f}")
                    print(f"F1-Score: {val_metrics['f1_score']:.4f}")
                    if 'auc_roc' in val_metrics:
                        print(f"AUC-ROC: {val_metrics['auc_roc']:.4f}")
                    if 'auc_pr' in val_metrics:
                        print(f"AUC-PR: {val_metrics['auc_pr']:.4f}")
            except:
                # Si predict_proba falla, mostrar métricas básicas
                if verbose:
                    acc = accuracy_score(y_val, y_val_pred)
                    f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
                    print("\nMétricas básicas en el conjunto de validación:")
                    print(f"Accuracy: {acc:.4f}")
                    print(f"F1-Score: {f1:.4f}")
        else:
            # Para modelos sin predict_proba
            if verbose:
                acc = accuracy_score(y_val, y_val_pred)
                f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
                print("\nMétricas básicas en el conjunto de validación:")
                print(f"Accuracy: {acc:.4f}")
                print(f"F1-Score: {f1:.4f}")
    
    return best_params, best_model, results