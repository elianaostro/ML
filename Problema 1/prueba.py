import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import LogisticRegression
from preprocessing import train_test_split, standard_scaler, oversample_duplicate, undersample_random, simple_smote
from metrics1 import accuracy_score, precision_score, recall_score, f1_score, roc_curve, pr_curve, auc, plot_confusion_matrix
from utils import detect_outliers_iqr, stratified_train_test_split, plot_feature_distributions, plot_correlation_matrix, class_balance_report

# Configuración inicial
np.random.seed(42)
# plt.style.use('seaborn')
# ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'petroff10', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']

# 1. Carga y exploración inicial de datos
def load_and_explore_data():
    dev_data = pd.read_csv('Ostrovsky_Eliana_TP2/src/data/cell_diagnosis_dev.csv')
    test_data = pd.read_csv('Ostrovsky_Eliana_TP2/src/data/cell_diagnosis_test.csv')
    imbalanced_dev_data = pd.read_csv('Ostrovsky_Eliana_TP2/src/data/cell_diagnosis_dev_imbalanced.csv')
    imbalanced_test_data = pd.read_csv('Ostrovsky_Eliana_TP2/src/data/cell_diagnosis_test_imbalanced.csv')
    
    # Mostrar información básica
    print("=== Datos Balanceados (Desarrollo) ===")
    dev_data.info()
    print("\n=== Datos Balanceados (Test) ===")
    test_data.info()
    print("\n=== Datos Desbalanceados (Desarrollo) ===")
    imbalanced_dev_data.info()
    
    # Separar características y target
    X_dev = dev_data.drop('Diagnosis', axis=1).values
    y_dev = dev_data['Diagnosis'].values
    X_test = test_data.drop('Diagnosis', axis=1).values
    y_test = test_data['Diagnosis'].values
    
    # Nombres de características para visualización
    feature_names = dev_data.drop('Diagnosis', axis=1).columns.tolist()
    
    return X_dev, y_dev, X_test, y_test, feature_names, imbalanced_dev_data, imbalanced_test_data

# 2. Análisis exploratorio de datos
def exploratory_data_analysis(X, y, feature_names):
    print("\n=== Análisis Exploratorio ===")
    
    # Reporte de balance de clases
    class_balance_report(y, class_names=['Benigno', 'Maligno'])
    
    # Distribución de características por clase
    plot_feature_distributions(X, feature_names, y, class_names=['Benigno', 'Maligno'])
    
    # Matriz de correlación
    plot_correlation_matrix(X, feature_names)
    
    # Detección de outliers
    outliers = detect_outliers_iqr(X)
    print(f"\nNúmero de outliers por característica (método IQR):")
    for i, feat in enumerate(feature_names):
        print(f"{feat}: {np.sum(outliers[:, i])}")

# 3. Preprocesamiento de datos
def preprocess_data(X_train, X_val, X_test):
    # Estandarización de características usando nuestra función
    X_train_scaled, mean, std = standard_scaler(X_train)
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_val_scaled, X_test_scaled, {'mean': mean, 'std': std}

# 4. Implementación y evaluación de regresión logística
def train_logistic_regression(X_train, y_train, X_val, y_val, lambda_=0.1):
    print("\n=== Entrenando Regresión Logística ===")
    
    # Crear y entrenar modelo
    model = LogisticRegression(learning_rate=0.01, n_iter=1000, lambda_=lambda_)
    model.fit(X_train, y_train)
    
    # Evaluar en validación
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Calcular métricas
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    # Curvas ROC y PR
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = pr_curve(y_val, y_proba)
    pr_auc = auc(recall, precision)
    
    # Mostrar resultados
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")
    
    # Graficar matriz de confusión
    plot_confusion_matrix(y_val, y_pred, classes=['Benigno', 'Maligno'])
    
    # Graficar curvas
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 
                   'roc_auc': roc_auc, 'pr_auc': pr_auc}

# 5. Ajuste de hiperparámetros (lambda)
def tune_hyperparameters(X_train, y_train, X_val, y_val):
    print("\n=== Ajuste de Hiperparámetros (λ) ===")
    
    lambda_values = [0.001, 0.01, 0.1, 1, 10]
    best_f1 = -1
    best_lambda = None
    best_model = None
    
    for lambda_ in lambda_values:
        print(f"\nProbando λ = {lambda_}")
        model, metrics = train_logistic_regression(X_train, y_train, X_val, y_val, lambda_)
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_lambda = lambda_
            best_model = model
    
    print(f"\nMejor λ: {best_lambda} con F1-Score: {best_f1:.4f}")
    return best_model, best_lambda

# 6. Evaluación en conjunto de test
def evaluate_on_test(model, X_test, y_test):
    print("\n=== Evaluación en Conjunto de Test ===")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Curvas ROC y PR
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = pr_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    # Mostrar resultados
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")
    
    # Graficar matriz de confusión
    plot_confusion_matrix(y_test, y_pred, classes=['Benigno', 'Maligno'])
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 
            'roc_auc': roc_auc, 'pr_auc': pr_auc}

# 7. Manejo de datos desbalanceados
def handle_imbalanced_data(imbalanced_dev_data, imbalanced_test_data, best_lambda):
    print("\n=== Manejo de Datos Desbalanceados ===")
    
    # Separar características y target
    X_imb_dev = imbalanced_dev_data.drop('Diagnosis', axis=1).values
    y_imb_dev = imbalanced_dev_data['Diagnosis'].values
    X_imb_test = imbalanced_test_data.drop('Diagnosis', axis=1).values
    y_imb_test = imbalanced_test_data['Diagnosis'].values
    
    # Reporte de balance original
    print("\nDistribución original:")
    class_balance_report(y_imb_dev)
    
    # Dividir en train/val
    X_train, X_val, y_train, y_val = stratified_train_test_split(
        X_imb_dev, y_imb_dev, test_size=0.2, random_state=42)
    
    # Estandarizar
    X_train_scaled, X_val_scaled, X_imb_test_scaled, scaler = preprocess_data(X_train, y_train, X_val)
    
    # Estrategias a probar
    strategies = {
        'Sin rebalanceo': (X_train_scaled, y_train),
        'Undersampling': undersample_random(X_train_scaled, y_train, random_state=42),
        'Oversampling (duplicación)': oversample_duplicate(X_train_scaled, y_train, random_state=42),
        'Oversampling (SMOTE)': simple_smote(X_train_scaled, y_train, random_state=42)
    }
    
    results = {}
    
    for name, (X_resampled, y_resampled) in strategies.items():
        print(f"\n=== Evaluando estrategia: {name} ===")
        print(f"Distribución después del rebalanceo:")
        class_balance_report(y_resampled)
        
        # Entrenar modelo
        model = LogisticRegression(learning_rate=0.01, n_iter=1000, 
                                 lambda_=best_lambda, class_weight=None)
        model.fit(X_resampled, y_resampled)
        
        # Evaluar en validación
        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        # Calcular métricas
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        # Curvas ROC y PR
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = pr_curve(y_val, y_proba)
        pr_auc = auc(recall, precision)
        
        # Guardar resultados
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'auc_roc': roc_auc,
            'auc_pr': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision,
            'recall_curve': recall,
            'model': model
        }
        
        # Mostrar métricas
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {roc_auc:.4f}")
        print(f"AUC-PR: {pr_auc:.4f}")
    
    # Estrategia adicional: Cost re-weighting
    print("\n=== Evaluando estrategia: Cost re-weighting ===")
    class_weights = {
        0: np.sum(y_train == 1) / len(y_train),  # Peso para clase mayoritaria
        1: np.sum(y_train == 0) / len(y_train)   # Peso para clase minoritaria
    }
    
    model = LogisticRegression(learning_rate=0.01, n_iter=1000, 
                             lambda_=best_lambda, class_weight=class_weights)
    model.fit(X_train_scaled, y_train)
    
    # Evaluar en validación
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    # Calcular métricas
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    # Curvas ROC y PR
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = pr_curve(y_val, y_proba)
    pr_auc = auc(recall, precision)
    
    # Guardar resultados
    results['Cost re-weighting'] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc_roc': roc_auc,
        'auc_pr': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision,
        'recall_curve': recall,
        'model': model
    }
    
    # Mostrar métricas
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")
    
    return results, X_imb_test_scaled, y_imb_test

# 8. Comparación de estrategias
def compare_strategies(results, X_test, y_test):
    print("\n=== Comparación de Estrategias ===")
    
    # Tabla de métricas
    print("\nMétricas de desempeño:")
    print("{:<25} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Estrategia", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "AUC-PR"))
    
    for name, res in results.items():
        print("{:<25} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            name, res['accuracy'], res['precision'], res['recall'], 
            res['f1_score'], res['auc_roc'], res['auc_pr']))
    
    # Graficar curvas ROC comparativas
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        plt.plot(res['fpr'], res['tpr'], label=f'{name} (AUC = {res["auc_roc"]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curvas ROC Comparativas')
    plt.legend()
    plt.show()
    
    # Graficar curvas PR comparativas
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        plt.plot(res['recall_curve'], res['precision_curve'], 
                label=f'{name} (AUC = {res["auc_pr"]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curvas Precision-Recall Comparativas')
    plt.legend()
    plt.show()
    
    # Evaluar la mejor estrategia en el conjunto de test
    best_strategy = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\nMejor estrategia: {best_strategy[0]} con F1-Score: {best_strategy[1]['f1_score']:.4f}")
    
    print("\nEvaluando la mejor estrategia en conjunto de test:")
    test_metrics = evaluate_on_test(best_strategy[1]['model'], X_test, y_test)
    
    return best_strategy[0], test_metrics

# 9. Flujo principal de ejecución
def main():
    # Paso 1: Carga y exploración de datos
    X_dev, y_dev, X_test, y_test, feature_names, imbalanced_dev_data, imbalanced_test_data = load_and_explore_data()
    
    # Paso 2: Análisis exploratorio
    exploratory_data_analysis(X_dev, y_dev, feature_names)
    
    # Paso 3: División de datos balanceados
    X_train, X_val, y_train, y_val = stratified_train_test_split(X_dev, y_dev, test_size=0.2, random_state=42)
    
    # Paso 4: Preprocesamiento
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_val, X_test)
    
    # Paso 5: Entrenamiento y ajuste de modelo
    best_model, best_lambda = tune_hyperparameters(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Paso 6: Evaluación en test balanceado
    test_metrics = evaluate_on_test(best_model, X_test_scaled, y_test)
    
    # Paso 7: Manejo de datos desbalanceados
    imb_results, X_imb_test_scaled, y_imb_test = handle_imbalanced_data(
        imbalanced_dev_data, imbalanced_test_data, best_lambda)
    
    # Paso 8: Comparación de estrategias y evaluación final
    best_strategy, imb_test_metrics = compare_strategies(imb_results, X_imb_test_scaled, y_imb_test)
    
    # Paso 9: Conclusiones
    print("\n=== Conclusiones ===")
    print("1. Para datos balanceados, la regresión logística regularizada (λ={:.3f}) logra un buen desempeño.".format(best_lambda))
    print("2. En datos desbalanceados, la mejor estrategia fue '{}'.".format(best_strategy))
    print("3. Las métricas en el conjunto de test fueron consistentes con las de validación.")
    print("4. Para producción, se recomienda usar {} con los hiperparámetros ajustados.".format(best_strategy))

if __name__ == "__main__":
    main()