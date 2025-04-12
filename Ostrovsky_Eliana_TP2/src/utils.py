import numpy as np
import matplotlib.pyplot as plt
from metrics import accuracy_score
import pandas as pd

def class_balance_report(y, class_names=None):
    """Genera un reporte del balance de clases."""
    y = y.astype(str)  # Convert all elements to strings to avoid type comparison issues
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    if class_names is None:
        class_names = [str(cls) for cls in classes]
    elif len(class_names) != len(classes):
        raise ValueError("La longitud de class_names no coincide con el número de clases únicas")
    
    # Crear mapeo de clase a nombre
    class_map = {cls: name for cls, name in zip(classes, class_names)}
    
    report = {
        'counts': {class_map[cls]: count for cls, count in zip(classes, counts)},
        'proportions': {class_map[cls]: count/total for cls, count in zip(classes, counts)},
        'total_samples': total
    }
    
    print("Reporte de Balance de Clases:")
    print(f"Total de muestras: {total}")
    print("\nConteo por clase:")
    for cls, count in report['counts'].items():
        print(f"- {cls}: {count} muestras ({report['proportions'][cls]:.2%})")
    
    return report

def apply_cost_sensitive_weights(y, class_weights='balanced'):
    """
    Calcula pesos para muestras según el balance de clases.
    
    Parámetros:
    - y: Etiquetas de clase
    - class_weights: 'balanced' para pesos inversamente proporcionales a frecuencia de clase,
                    o dict con pesos personalizados {0: w0, 1: w1}
    
    Retorna:
    - Array de pesos para cada muestra
    """

    if isinstance(class_weights, dict):
        return np.array([class_weights[cls] for cls in y])
    
    # Calcular pesos balanceados
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    
    weights = n_samples / (n_classes * counts)
    weight_map = {cls: weight for cls, weight in zip(classes, weights)}
    
    return np.array([weight_map[cls] for cls in y])

def plot_decision_boundary(model, X, y, title="Decision Boundary", step=0.02, figsize=(10, 6)):
    """
    Grafica la frontera de decisión de un modelo 2D.
    
    Parámetros:
    - model: Modelo entrenado con métodos predict() y predict_proba()
    - X: Datos de características (debe tener exactamente 2 características)
    - y: Etiquetas verdaderas
    - title: Título del gráfico
    - step: Paso para la malla de predicción
    - figsize: Tamaño de la figura
    """
    if X.shape[1] != 2:
        raise ValueError("Esta función solo funciona para datos 2D")
    
    # Crear malla para predicción
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    
    # Predecir para cada punto de la malla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Graficar
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5, random_state=None):
    """
    Genera curva de aprendizaje para un modelo.
    
    Parámetros:
    - model: Modelo a evaluar
    - X: Datos de características
    - y: Etiquetas
    - train_sizes: Proporciones del conjunto de entrenamiento a evaluar
    - cv: Número de divisiones para validación cruzada
    - random_state: Semilla para reproducibilidad
    
    Retorna:
    - train_scores: Puntajes de entrenamiento para cada tamaño
    - val_scores: Puntajes de validación para cada tamaño
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    train_sizes_abs = (train_sizes * n_samples).astype(int)
    
    train_scores = []
    val_scores = []
    
    for size in train_sizes_abs:
        fold_train_scores = []
        fold_val_scores = []
        
        for _ in range(cv):
            # Crear división aleatoria
            indices = np.random.permutation(n_samples)
            train_idx, val_idx = indices[:size], indices[size:]
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Entrenar y evaluar
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            fold_train_scores.append(accuracy_score(y_train, train_pred))
            
            val_pred = model.predict(X_val)
            fold_val_scores.append(accuracy_score(y_val, val_pred))
        
        train_scores.append(np.mean(fold_train_scores))
        val_scores.append(np.mean(fold_val_scores))
    
    # Graficar curva de aprendizaje
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes_abs, train_scores, 'o-', label="Training score")
    plt.plot(train_sizes_abs, val_scores, 'o-', label="Validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid()
    plt.show()
    
    return train_scores, val_scores

def feature_importance_plot(model, feature_names, title="Feature Importance"):
    """
    Grafica la importancia de características para modelos basados en árboles.
    
    Parámetros:
    - model: Modelo entrenado (RandomForest o DecisionTree)
    - feature_names: Lista de nombres de características
    - title: Título del gráfico
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("El modelo no tiene atributo feature_importances_")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.show()

def save_model(model, filename):
    """
    Guarda un modelo en disco usando pickle.
    
    Parámetros:
    - model: Modelo a guardar
    - filename: Nombre del archivo (incluyendo extensión .pkl)
    """
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """
    Carga un modelo desde disco.
    
    Parámetros:
    - filename: Nombre del archivo (incluyendo extensión .pkl)
    
    Retorna:
    - Modelo cargado
    """
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)