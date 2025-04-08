import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from metrics import accuracy_score

def stratified_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    División estratificada de datos en conjuntos de entrenamiento y prueba.
    Mantiene la proporción de clases en ambos conjuntos.
    
    Parámetros:
    - X: Datos de características (numpy array)
    - y: Etiquetas (numpy array)
    - test_size: Proporción del conjunto de prueba (0-1)
    - random_state: Semilla para reproducibilidad
    
    Retorna:
    - X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in classes}
    
    train_indices = []
    test_indices = []
    
    for cls in classes:
        indices = class_indices[cls]
        np.random.shuffle(indices)
        
        n_test = int(len(indices) * test_size)
        
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])
    
    # Mezclar los índices para evitar orden por clase
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def plot_feature_distributions(X, feature_names, y, class_names=None, n_cols=3, figsize=(15, 10)):
    """
    Grafica distribuciones de características por clase.
    
    Parámetros:
    - X: Datos de características (numpy array)
    - feature_names: Lista de nombres de características
    - y: Etiquetas de clase
    - class_names: Lista de nombres de clases
    - n_cols: Número de columnas en la cuadrícula de gráficos
    - figsize: Tamaño de la figura
    """
    if class_names is None:
        class_names = np.unique(y)
    
    n_features = X.shape[1]
    n_rows = int(np.ceil(n_features / n_cols))
    
    plt.figure(figsize=figsize)
    
    for i in range(n_features):
        plt.subplot(n_rows, n_cols, i+1)
        
        for cls in np.unique(y):
            plt.hist(X[y == cls, i], alpha=0.5, label=f'Class {class_names[cls]}')
        
        plt.title(feature_names[i])
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(X, feature_names, figsize=(10, 8)):
    """
    Grafica matriz de correlación entre características.
    
    Parámetros:
    - X: Datos de características (numpy array)
    - feature_names: Lista de nombres de características
    - figsize: Tamaño de la figura
    """
    corr_matrix = np.corrcoef(X, rowvar=False)
    
    plt.figure(figsize=figsize)
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    
    plt.xticks(np.arange(len(feature_names)), feature_names, rotation=90)
    plt.yticks(np.arange(len(feature_names)), feature_names)
    
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            plt.text(j, i, f"{corr_matrix[i, j]:.2f}", 
                     ha="center", va="center", color="white")
    
    plt.title("Matriz de Correlación")
    plt.tight_layout()
    plt.show()

def detect_outliers_iqr(X, factor=1.5):
    """
    Detecta outliers usando el método del rango intercuartílico (IQR).
    
    Parámetros:
    - X: Datos de características (numpy array)
    - factor: Factor multiplicador del IQR (típicamente 1.5)
    
    Retorna:
    - Máscara booleana de outliers (True = outlier)
    """
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return (X < lower_bound) | (X > upper_bound)

def class_balance_report(y, class_names=None):
    """
    Genera un reporte del balance de clases.
    
    Parámetros:
    - y: Etiquetas de clase
    - class_names: Lista de nombres de clases
    
    Retorna:
    - Diccionario con conteos y proporciones por clase
    """
    if class_names is None:
        class_names = np.unique(y)
    
    counts = Counter(y)
    total = len(y)
    
    report = {
        'counts': {class_names[cls]: count for cls, count in counts.items()},
        'proportions': {class_names[cls]: count/total for cls, count in counts.items()},
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