#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para ejecutar la predicción de rendimiento de jugadores de basketball.

Este script orquesta todo el flujo de trabajo del análisis, desde la carga de datos
hasta la evaluación final y selección del mejor modelo.

Fecha: 11 de abril de 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# Asegurar que podamos importar desde la carpeta src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar funciones desde la carpeta src
from src.models import (LogisticRegression, MultinomialLogisticRegression, 
                       LDA, RandomForest)
from metrics1 import (confusion_matrix, accuracy_score, precision_score, 
                        recall_score, f1_score, roc_curve, pr_curve, auc, 
                        plot_confusion_matrix, report_metrics, display_metrics)

# Configuración de visualización y reproducibilidad
np.random.seed(42)
plt.style.use('ggplot')
sns.set(style='whitegrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# Constantes
DATA_PATH = 'data'
DEV_FILE = 'WAR_class_dev.csv'
TEST_FILE = 'WAR_class_test.csv'
RESULTS_PATH = 'results'
TARGET_COL = 'war_class'
CLASS_NAMES = {1: 'Negative WAR', 2: 'Null WAR', 3: 'Positive WAR'}

# Crear carpeta de resultados si no existe
os.makedirs(RESULTS_PATH, exist_ok=True)

def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    print(f"Cargando datos desde {file_path}...")
    data = pd.read_csv(file_path)
    print(f"Datos cargados. Forma: {data.shape}")
    return data

def explore_data(data, title="Análisis Exploratorio de Datos"):
    """
    Realiza un análisis exploratorio de los datos y guarda las visualizaciones.
    
    Args:
        data: DataFrame con los datos a analizar
        title: Título para las visualizaciones
    """
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    
    # Información básica
    print("\nInformación básica:")
    print(f"  Número de muestras: {data.shape[0]}")
    print(f"  Número de características: {data.shape[1]}")
    
    # Verificar valores faltantes
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print("\nValores faltantes:")
        print(missing[missing > 0])
    else:
        print("\nNo hay valores faltantes en los datos.")
    
    # Verificar duplicados
    duplicates = data.duplicated().sum()
    print(f"\nFilas duplicadas: {duplicates}")
    
    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas:")
    print(data.describe())
    
    # Distribución de clases
    if TARGET_COL in data.columns:
        class_counts = data[TARGET_COL].value_counts()
        print("\nDistribución de clases:")
        print(class_counts)
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x=data[TARGET_COL])
        plt.title('Distribución de Clases WAR')
        plt.xlabel('Clase WAR')
        plt.xticks([0, 1, 2], ['Negative WAR', 'Null WAR', 'Positive WAR'])
        plt.ylabel('Número de Jugadores')
        plt.savefig(os.path.join(RESULTS_PATH, 'class_distribution.png'), dpi=300)
        plt.close()
        
        # Calcular balance/desbalance
        total = len(data)
        print("\nBalance de clases:")
        for cls, count in class_counts.items():
            print(f"  Clase {cls} ({CLASS_NAMES[cls]}): {count} ({count/total:.2%})")
    
    # Correlación entre variables
    corr = data.corr()
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'correlation_matrix.png'), dpi=300)
    plt.close()
    
    # Identificar correlaciones fuertes
    strong_corr = (corr.abs() > 0.75) & (corr != 1.0)
    if strong_corr.any().any():
        print("\nVariables con alta correlación (|corr| > 0.75):")
        for col in corr.columns:
            strong_cols = corr.index[strong_corr[col]].tolist()
            for strong_col in strong_cols:
                if strong_col != col:  # Evitar mostrar correlación consigo mismo
                    print(f"  {col} - {strong_col}: {corr.loc[col, strong_col]:.3f}")
    
    # Distribución de características numéricas
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    num_cols = [col for col in num_cols if col != TARGET_COL]
    
    if num_cols:
        for i in range(0, len(num_cols), 3):
            cols_subset = num_cols[i:i+3]
            if cols_subset:
                plt.figure(figsize=(15, 5))
                for j, col in enumerate(cols_subset):
                    plt.subplot(1, len(cols_subset), j+1)
                    sns.histplot(data[col], kde=True)
                    plt.title(f'Distribución de {col}')
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_PATH, f'features_dist_{i}.png'), dpi=300)
                plt.close()
        
        # Boxplots por clase para las 5 características más correlacionadas con el target
        if TARGET_COL in data.columns:
            target_corr = corr[TARGET_COL].abs().sort_values(ascending=False)
            top_features = target_corr[1:6].index.tolist()  # Excluir el propio target
            
            plt.figure(figsize=(15, 10))
            for i, feature in enumerate(top_features):
                plt.subplot(2, 3, i+1)
                sns.boxplot(x=data[TARGET_COL], y=data[feature])
                plt.title(f'{feature} por Clase WAR')
                plt.xlabel('Clase WAR')
                plt.xticks([0, 1, 2], ['Negative', 'Null', 'Positive'])
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_PATH, 'top_features_by_class.png'), dpi=300)
            plt.close()

def preprocess_data(data):
    """
    Preprocesa los datos para modelado.
    
    Args:
        data: DataFrame con los datos a preprocesar
        
    Returns:
        X: Características normalizadas
        y: Variable objetivo
        scaler: Objeto StandardScaler ajustado
        feature_names: Nombres de las características
    """
    print("\nPreprocesando datos...")
    
    # Separar features y target
    if TARGET_COL in data.columns:
        X = data.drop(columns=[TARGET_COL])
        y = data[TARGET_COL]
    else:
        X = data
        y = None
    
    feature_names = X.columns.tolist()
    
    # Normalizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Preprocesamiento completado. X shape: {X_scaled.shape}")
    
    return X_scaled, y, scaler, feature_names

def train_and_evaluate_models(X_train, y_train, X_val, y_val):
    """
    Entrena y evalúa los tres modelos solicitados.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Target de entrenamiento
        X_val: Características de validación
        y_val: Target de validación
        
    Returns:
        dict: Diccionario con los modelos entrenados
        dict: Diccionario con las métricas de cada modelo
    """
    print("\n=====