"""
Predicción de Rendimiento de Jugadores de Basketball

Este script implementa un análisis completo sobre datos de jugadores de basketball,
utilizando diferentes modelos de clasificación para predecir la clase WAR (Wins Above Replacement).

Autor: [Tu Nombre]
Fecha: 11 de abril de 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from collections import Counter

# Importar módulos personalizados
from src.models import LDA, MultinomialLogisticRegression, RandomForest
from metrics1 import (confusion_matrix, accuracy_score, precision_score, 
                        recall_score, f1_score, roc_curve, pr_curve, auc, 
                        plot_confusion_matrix, report_metrics, display_metrics)

# Configuración
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

# Constantes
DATA_DEV_PATH = 'Ostrovsky_Eliana_TP2/data2/WAR_class_dev.csv'
DATA_TEST_PATH = 'Ostrovsky_Eliana_TP2/data2/WAR_class_test.csv'
TARGET_COLUMN = 'war_class'
WAR_CLASS_NAMES = {1: 'Negative WAR', 2: 'Null WAR', 3: 'Positive WAR'}

def load_data(filepath):
    """Carga los datos desde un archivo CSV."""
    print(f"Cargando datos desde {filepath}...")
    data = pd.read_csv(filepath)
    return data

def check_data_quality(data):
    """Verifica la calidad de los datos."""
    print("\n===== CALIDAD DE LOS DATOS =====")
    
    # Verificar valores faltantes
    missing_values = data.isnull().sum()
    print(f"Valores faltantes:\n{missing_values[missing_values > 0]}")
    
    # Verificar duplicados
    duplicates = data.duplicated().sum()
    print(f"Filas duplicadas: {duplicates}")
    
    # Verificar desbalanceo de clases
    if TARGET_COLUMN in data.columns:
        class_distribution = data[TARGET_COLUMN].value_counts()
        print(f"Distribución de clases:\n{class_distribution}")
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x=data[TARGET_COLUMN])
        plt.title('Distribución de clases (WAR)')
        plt.xlabel('WAR Class')
        plt.ylabel('Cantidad')
        plt.xticks(ticks=[0, 1, 2], labels=['Negative WAR', 'Null WAR', 'Positive WAR'])
        plt.show()

def eda(data):
    """Análisis exploratorio de datos."""
    print("\n===== ANÁLISIS EXPLORATORIO DE DATOS =====")
    
    # Estadísticas descriptivas
    print("Estadísticas descriptivas:")
    print(data.describe())
    
    # Distribución de variables numéricas
    print("\nDistribución de variables numéricas:")
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != TARGET_COLUMN]
    
    if len(numerical_cols) > 0:
        plt.figure(figsize=(15, 12))
        for i, col in enumerate(numerical_cols[:9], 1):  # Mostrar máximo 9 variables
            plt.subplot(3, 3, i)
            sns.histplot(data[col], kde=True)
            plt.title(f'Distribución de {col}')
        plt.tight_layout()
        plt.show()
    
    # Correlación entre variables
    print("\nCorrelación entre variables:")
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación')
    plt.show()
    
    # Identificar correlaciones fuertes (|corr| > 0.75)
    strong_corr = (correlation_matrix.abs() > 0.75) & (correlation_matrix.abs() < 1.0)
    if strong_corr.any().any():
        print("Pares de variables con correlación fuerte (|corr| > 0.75):")
        for col in correlation_matrix.columns:
            strong_pairs = correlation_matrix.index[strong_corr[col]].tolist()
            for strong_col in strong_pairs:
                print(f"{col} - {strong_col}: {correlation_matrix.loc[strong_col, col]:.2f}")
    
    # Gráficos de dispersión para la variable objetivo y algunas características importantes
    if TARGET_COLUMN in data.columns and len(numerical_cols) > 0:
        # Seleccionar algunas características importantes (ejemplo: las 4 con mayor correlación absoluta con el target)
        target_corr = correlation_matrix[TARGET_COLUMN].abs().sort_values(ascending=False)
        top_features = target_corr[1:5].index.tolist()  # Excluir el propio target
        
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(top_features, 1):
            plt.subplot(2, 2, i)
            sns.boxplot(x=data[TARGET_COLUMN], y=data[feature])
            plt.title(f'{feature} por WAR Class')
            plt.xlabel('WAR Class')
        plt.tight_layout()
        plt.show()

def preprocess_data(data):
    """Preprocesa los datos para el modelado."""
    print("\n===== PREPROCESAMIENTO DE DATOS =====")
    
    # Separar características y variable objetivo
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    
    # Normalizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Forma de X: {X.shape}")
    print(f"Clases únicas en y: {np.unique(y)}")
    
    return X_scaled, y, scaler, X.columns

def train_evaluate_model(model_name, model, X_train, y_train, X_val, y_val, class_names=None):
    """Entrena y evalúa un modelo, devolviendo las métricas y el modelo entrenado."""
    print(f"\n===== ENTRENANDO MODELO: {model_name} =====")
    
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = model.predict(X_val)
    
    # Para métricas que requieren probabilidades, verificar si el modelo tiene predict_proba
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_val)
    else:
        # Si no tiene predict_proba, crear una matriz ficticia para mantener la compatibilidad
        n_classes = len(np.unique(y_train))
        y_proba = np.zeros((len(y_val), n_classes))
        for i, pred in enumerate(y_pred):
            y_proba[i, int(pred-1)] = 1  # Asumiendo que las clases comienzan en 1
    
    # Calcular y mostrar métricas
    print(f"Métricas de {model_name}:")
    metrics = report_metrics(y_val, y_pred, y_proba, np.unique(y_train))
    
    # Mostrar matriz de confusión
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_val, y_pred, np.unique(y_train))
    if class_names:
        class_labels = [class_names[cls] for cls in np.unique(y_train)]
    else:
        class_labels = [f'Clase {cls}' for cls in np.unique(y_train)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()
    
    # Mostrar curvas ROC y PR para clasificación multiclase (one-vs-rest)
    plt.figure(figsize=(15, 6))
    
    # Curva ROC
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = metrics['roc_curve']
    plt.plot(fpr, tpr, label=f'AUC = {metrics["auc_roc"]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend()
    
    # Curva PR
    plt.subplot(1, 2, 2)
    precision, recall, _ = metrics['pr_curve']
    plt.plot(recall, precision, label=f'AUC = {metrics["auc_pr"]:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall - {model_name}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resumen de métricas
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR: {metrics['auc_pr']:.4f}")
    
    return metrics, model

def evaluate_model_on_test(model, model_name, X_test, y_test, class_names=None):
    """Evalúa un modelo en el conjunto de test."""
    print(f"\n===== EVALUANDO MODELO EN TEST: {model_name} =====")
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Para métricas que requieren probabilidades
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    else:
        n_classes = len(np.unique(y_test))
        y_proba = np.zeros((len(y_test), n_classes))
        for i, pred in enumerate(y_pred):
            y_proba[i, int(pred-1)] = 1
    
    # Calcular métricas
    metrics = report_metrics(y_test, y_pred, y_proba, np.unique(y_test))
    
    # Mostrar matriz de confusión
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred, np.unique(y_test))
    if class_names:
        class_labels = [class_names[cls] for cls in np.unique(y_test)]
    else:
        class_labels = [f'Clase {cls}' for cls in np.unique(y_test)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(f'Matriz de Confusión (Test) - {model_name}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()
    
    # Imprimir resumen de métricas
    print(f"Accuracy (Test): {metrics['accuracy']:.4f}")
    print(f"Precision (Test): {metrics['precision']:.4f}")
    print(f"Recall (Test): {metrics['recall']:.4f}")
    print(f"F1-Score (Test): {metrics['f1_score']:.4f}")
    print(f"AUC-ROC (Test): {metrics['auc_roc']:.4f}")
    print(f"AUC-PR (Test): {metrics['auc_pr']:.4f}")
    
    return metrics

def compare_models(models_metrics, test_metrics=None):
    """Compara las métricas de diferentes modelos."""
    print("\n===== COMPARACIÓN DE MODELOS =====")
    
    # Crear un DataFrame para comparar métricas
    metrics_df = pd.DataFrame({
        model_name: {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'AUC-ROC': metrics['auc_roc'],
            'AUC-PR': metrics['auc_pr']
        }
        for model_name, metrics in models_metrics.items()
    })
    
    print("Comparación de métricas (validación):")
    print(metrics_df)
    
    # Visualizar comparación
    plt.figure(figsize=(12, 8))
    metrics_df.plot(kind='bar')
    plt.title('Comparación de Métricas entre Modelos')
    plt.ylabel('Valor')
    plt.xlabel('Métrica')
    plt.xticks(rotation=45)
    plt.legend(title='Modelo')
    plt.tight_layout()
    plt.show()
    
    # Si hay métricas de test, hacer una comparación entre validación y test
    if test_metrics:
        for model_name, metrics in test_metrics.items():
            val_metrics = models_metrics[model_name]
            
            comparison_df = pd.DataFrame({
                'Validación': {
                    'Accuracy': val_metrics['accuracy'],
                    'Precision': val_metrics['precision'],
                    'Recall': val_metrics['recall'],
                    'F1-Score': val_metrics['f1_score'],
                    'AUC-ROC': val_metrics['auc_roc'],
                    'AUC-PR': val_metrics['auc_pr']
                },
                'Test': {
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'AUC-ROC': metrics['auc_roc'],
                    'AUC-PR': metrics['auc_pr']
                }
            })
            
            print(f"\nComparación de métricas de validación vs. test para {model_name}:")
            print(comparison_df)
            
            # Visualizar comparación
            plt.figure(figsize=(10, 6))
            comparison_df.plot(kind='bar')
            plt.title(f'Comparación de Métricas: Validación vs. Test ({model_name})')
            plt.ylabel('Valor')
            plt.xlabel('Métrica')
            plt.xticks(rotation=45)
            plt.legend(title='Conjunto')
            plt.tight_layout()
            plt.show()

def main():
    """Función principal para ejecutar el análisis completo."""
    print("======= PREDICCIÓN DE RENDIMIENTO DE JUGADORES DE BASKETBALL =======")
    
    # 1. Cargar datos de desarrollo
    data_dev = load_data(DATA_DEV_PATH)
    
    # 2. Verificar calidad de datos
    check_data_quality(data_dev)
    
    # 3. Análisis exploratorio
    eda(data_dev)
    
    # 4. Preprocesar datos
    X, y, scaler, feature_names = preprocess_data(data_dev)
    
    # 5. División en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"División de datos: {X_train.shape[0]} muestras de entrenamiento, {X_val.shape[0]} muestras de validación")
    
    # 6. Entrenamiento y evaluación de modelos
    # Inicializar modelos
    models = {
        'LDA': LDA(),
        'Regresión Logística Multinomial': MultinomialLogisticRegression(learning_rate=0.01, n_iter=1000, lambda_=0.1),
        'Random Forest': RandomForest(n_estimators=100, max_depth=10, criterion='entropy', random_state=42)
    }
    
    # Entrenar y evaluar cada modelo
    trained_models = {}
    models_metrics = {}
    
    for model_name, model in models.items():
        metrics, trained_model = train_evaluate_model(
            model_name, model, X_train, y_train, X_val, y_val, WAR_CLASS_NAMES
        )
        models_metrics[model_name] = metrics
        trained_models[model_name] = trained_model
    
    # 7. Comparar modelos
    compare_models(models_metrics)
    
    # 8. Entrenar con todos los datos de desarrollo y evaluar en test
    print("\n===== ENTRENAMIENTO CON TODOS LOS DATOS DE DESARROLLO =====")
    data_test = load_data(DATA_TEST_PATH)
    X_test_raw = data_test.drop(columns=[TARGET_COLUMN])
    y_test = data_test[TARGET_COLUMN]
    
    # Asegurar que X_test tenga las mismas columnas que X_train (por si acaso)
    X_test = scaler.transform(X_test_raw)
    
    # Re-entrenar modelos con todos los datos de desarrollo
    retrained_models = {}
    for model_name, model in models.items():
        print(f"Re-entrenando {model_name} con todos los datos de desarrollo...")
        new_model = type(model)()  # Crear una nueva instancia del mismo tipo
        if model_name == 'LDA':
            new_model = LDA()
        elif model_name == 'Regresión Logística Multinomial':
            new_model = MultinomialLogisticRegression(learning_rate=0.01, n_iter=1000, lambda_=0.1)
        elif model_name == 'Random Forest':
            new_model = RandomForest(n_estimators=100, max_depth=10, criterion='entropy', random_state=42)
        
        new_model.fit(X, y)
        retrained_models[model_name] = new_model
    
    # 9. Evaluar en conjunto de test
    test_metrics = {}
    for model_name, model in retrained_models.items():
        metrics = evaluate_model_on_test(
            model, model_name, X_test, y_test, WAR_CLASS_NAMES
        )
        test_metrics[model_name] = metrics
    
    # 10. Comparar métricas de validación vs. test
    compare_models(models_metrics, test_metrics)
    
    # 11. Selección del mejor modelo
    print("\n===== SELECCIÓN DEL MEJOR MODELO =====")
    # Aquí puedes implementar una lógica personalizada para seleccionar el mejor modelo
    # Por ejemplo, basándote en el F1-score o en AUC-ROC
    best_model_name = max(test_metrics.items(), key=lambda x: x[1]['f1_score'])[0]
    print(f"El mejor modelo según F1-Score en test es: {best_model_name}")
    
    # Discusión sobre el modelo seleccionado
    print("\nDiscusión sobre el modelo seleccionado:")
    print(f"El modelo {best_model_name} presenta el mejor rendimiento general en el conjunto de test.")
    print("Ventajas del modelo seleccionado:")
    # Aquí se agregaría un análisis personalizado basado en los resultados obtenidos
    
    # 12. Conclusiones
    print("\n===== CONCLUSIONES =====")
    # Aquí agregarías tus conclusiones finales basadas en todo el análisis

if __name__ == "__main__":
    main()
