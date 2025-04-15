# Trabajo Práctico 2: Clasificación y Ensemble Learning

Este repositorio contiene la implementación del Trabajo Práctico 2 para la materia I302 - Aprendizaje Automático y Aprendizaje Profundo, 1er Semestre 2025.

## Estructura del Proyecto

```
├── data1/                            # Datos del problema 1: Diagnóstico de Cáncer de Mama
│   ├── cell_diagnosis_dev.csv        # Datos balanceados para desarrollo
│   ├── cell_diagnosis_test.csv       # Datos balanceados para test
│   ├── cell_diagnosis_dev_imbalanced.csv # Datos desbalanceados para desarrollo
│   ├── cell_diagnosis_test_imbalanced.csv # Datos desbalanceados para test
│   └── cell_diagnosis_description.md # Descripción detallada de las variables
│
├── data2/                            # Datos del problema 2: Predicción de Rendimiento de Jugadores de Basketball
│   ├── WAR_class_dev.csv             # Datos de desarrollo
│   ├── WAR_class_test.csv            # Datos de test
│   └── WAR_class.md                  # Descripción de las variables (poss, mp, off_def, pace_impact)
│
├── notebooks/                        # Jupyter notebooks con soluciones
│   ├── ostrovsky-eliana-notebook-tp2-p1.ipynb  # Solución al problema 1
│   └── ostrovsky-eliana-notebook-tp2-p2.ipynb  # Solución al problema 2
│
├── modelos/                          # Modelos entrenados guardados
│   ├── Diagnostico_balance_model.pkl   # Modelo de regresión logística para datos balanceados
│   ├── Diagnostico_imbalance_model.pkl # Modelo de regresión logística para datos desbalanceados
│   ├── War_LDA_model.pkl               # Modelo LDA para predicción de rendimiento
│   ├── War_MLR_model.pkl               # Modelo de regresión logística multinomial
│   └── War_RF_model.pkl                # Modelo Random Forest
│
├── src/                              # Código fuente modularizado
│   ├── balanced.py                   # Implementación de técnicas de rebalanceo
│   ├── metrics.py                    # Funciones para métricas de evaluación
│   ├── models.py                     # Implementación de modelos de ML desde cero
│   ├── plots.py                      # Funciones para visualización
│   ├── preprocessing.py              # Funciones de preprocesamiento
│   └── utils.py                      # Funciones auxiliares
│
├── intro.html                        # Página de introducción al proyecto
├── README.md                         # Documentación principal
├── requirements.txt                  # Dependencias del proyecto
└── ostrovsky-eliana-informe-tp2.pdf  # Informe del trabajo práctico
```

## Problema 1: Diagnóstico de Cáncer de Mama

Implementación de un modelo de regresión logística binaria con regularización L2 para la clasificación de tumores de mama (benignos o malignos). Se exploraron distintas técnicas de rebalanceo para mejorar la detección en conjuntos de datos desbalanceados.

El conjunto de datos incluye variables morfológicas y moleculares de células, como tamaño celular, forma, densidad nuclear, tasa de mitosis y presencia de mutaciones genéticas. La descripción completa se encuentra en `cell_diagnosis_description.md`.

## Problema 2: Predicción de Rendimiento de Jugadores de Basketball

Implementación de tres algoritmos de clasificación multiclase para predecir el rendimiento de jugadores de basketball:
- Análisis Discriminante Lineal (LDA)
- Regresión Logística Multinomial
- Random Forest

Las variables utilizadas incluyen el número de posesiones, minutos jugados, impacto en ofensa/defensa y velocidad de juego. El objetivo es clasificar a los jugadores en tres categorías: Negative WAR (1), Null WAR (2) o Positive WAR (3). Ver `WAR_class.md` para más detalles.

## Implementaciones desde Cero

Todos los algoritmos fueron implementados desde cero utilizando NumPy, sin depender de bibliotecas de machine learning como scikit-learn.

## Requisitos

Para ejecutar este proyecto necesitas instalar las dependencias listadas en `requirements.txt`:

```
pip install -r requirements.txt
```

## Instrucciones de Uso

1. Asegúrate de tener todas las dependencias instaladas
2. Los notebooks en la carpeta `notebooks/` contienen la solución paso a paso
3. Puedes importar los modelos entrenados desde la carpeta `modelos/`

## Autor

Eliana Ostrovsky