# Aprendizaje Automatico y Aprendizaje Profundo (I302)

Repositorio de trabajos practicos y material de tutoriales para la materia I302 - Aprendizaje Automatico y Aprendizaje Profundo, 1er Semestre 2025.

## Estructura del Repositorio

```
├── Ostrovsky_Eliana_TP1/       # TP1: Regresion Lineal
├── Ostrovsky_Eliana_TP2/       # TP2: Clasificacion y Ensemble Learning
├── Ostrovsky_Eliana_TP3/       # TP3: Redes Neuronales
├── Ostrovsky_Eliana_TP4/       # TP4: Aprendizaje No Supervisado
└── Clases Tutoriales/          # Material de clases tutoriales (1-10)
```

## Trabajos Practicos

### TP1: Regresion Lineal

Prediccion de precios de casas utilizando regresion lineal implementada desde cero.

- Regresion lineal con pseudoinversa y descenso por gradiente
- Feature engineering: variables polinomiales, interacciones, distancias geograficas
- Regularizacion L2 (Ridge) y L1 (Lasso)
- Optimizacion de hiperparametros con validacion cruzada
- Preprocesamiento: imputacion de valores faltantes con KNN, conversion de unidades

**Tecnologias:** Python, NumPy, Pandas, Matplotlib

### TP2: Clasificacion y Ensemble Learning

Dos problemas de clasificacion resueltos con algoritmos implementados desde cero.

**Problema 1 - Diagnostico de Cancer de Mama:**
- Regresion logistica binaria con regularizacion L2
- Manejo de datos desbalanceados (SMOTE, oversampling, undersampling)

**Problema 2 - Prediccion de Rendimiento de Jugadores de Basketball:**
- Analisis Discriminante Lineal (LDA)
- Regresion Logistica Multinomial
- Random Forest

**Tecnologias:** Python, NumPy, Pandas, Matplotlib

### TP3: Redes Neuronales

Clasificacion de caracteres japoneses (49 clases, imagenes 28x28) con redes neuronales.

- Red neuronal con backpropagation implementada desde cero en NumPy
- Mejoras: learning rate scheduling, mini-batch SGD, ADAM, dropout, batch normalization, early stopping
- Implementacion comparativa en PyTorch
- Analisis de overfitting y exploracion de arquitecturas

**Modelos:**
| Modelo | Descripcion |
|--------|-------------|
| M0 | Implementacion basica (2 capas ocultas: 100, 80 neuronas) |
| M1 | Mejor configuracion con mejoras avanzadas |
| M2 | PyTorch con misma arquitectura que M1 |
| M3 | Mejor arquitectura en PyTorch |
| M4 | Modelo PyTorch con overfitting intencional |

**Tecnologias:** Python, NumPy, PyTorch, Matplotlib

### TP4: Aprendizaje No Supervisado

Clustering y reduccion de dimensionalidad sobre datasets sinteticos y MNIST.

**Clustering:**
- K-Means
- Gaussian Mixture Models (GMM)
- DBSCAN
- Metodo del codo para seleccion de clusters

**Reduccion de Dimensionalidad:**
- PCA (Principal Component Analysis)
- Analisis de error de reconstruccion y varianza explicada
- Visualizacion 2D de datos de alta dimensionalidad

**Tecnologias:** Python, NumPy, Pandas, Matplotlib, scikit-learn

## Clases Tutoriales

| Tutorial | Tema |
|----------|------|
| 1 | Introduccion al aprendizaje automatico |
| 2 | Regresion Lineal (NumPy y Matplotlib) |
| 3 | Feature Engineering |
| 4 | Regresion Localmente Ponderada |
| 6 | Arboles de Decision |
| 7 | MLPs (Perceptrones Multicapa) |
| 8 | Introduccion a PyTorch y TensorBoard |
| 9 | t-SNE |
| 10 | MLflow |

## Tecnologias Utilizadas

- **Python** - Lenguaje principal
- **NumPy** - Implementaciones from scratch de algoritmos de ML
- **Pandas** - Manipulacion y analisis de datos
- **Matplotlib** - Visualizacion
- **PyTorch** - Deep learning (TP3)
- **MLflow** - Tracking de experimentos (Tutorial 10)
- **Jupyter Notebooks** - Desarrollo interactivo

## Autor

Eliana Ostrovsky
