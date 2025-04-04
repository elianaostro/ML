import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset (reemplazar 'dataset.csv' con el nombre real)
df = pd.read_csv('Ostrovsky_Eliana_TP2/Problema 1/Data/raw/cell_diagnosis_dev.csv')

# Mostrar información general del dataset
print(df.info())
print(df.describe())

# --- LIMPIEZA DE DATOS --- #
# Corregir errores ortográficos en variables categóricas
df['CellType'] = df['CellType'].replace({'Epthlial': 'Epithelial', 'Mesnchymal': 'Mesenchymal'})
df['GeneticMutation'] = df['GeneticMutation'].replace({'Presnt': 'Present', 'Absnt': 'Absent'})

# Imputar valores nulos en variables categóricas con 'Desconocido'
df['CellType'] = df['CellType'].fillna('Unknown')

# Imputar valores nulos en variables numéricas con la mediana
num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    median_value = np.nanmedian(df[col])
    df[col] = np.where(np.isnan(df[col]), median_value, df[col])

# Eliminar valores atípicos (truncar a percentiles 1% y 99%)
for col in num_cols:
    p1, p99 = np.percentile(df[col], [1, 99])
    df[col] = np.clip(df[col], p1, p99)

# --- ANÁLISIS EXPLORATORIO --- #
# Distribución del diagnóstico
plt.figure(figsize=(6,4))
sns.countplot(x='Diagnosis', data=df, hue='Diagnosis', palette='coolwarm', dodge=False, legend=False)
plt.title('Distribución de Diagnóstico')
plt.show()

# Matriz de correlación
plt.figure(figsize=(10, 6))
corr_matrix = np.corrcoef(df[num_cols].T)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=num_cols, yticklabels=num_cols)
plt.title('Matriz de Correlación')
plt.show()

# Comparación de medias entre células normales y anómalas
df_grouped = df.groupby('Diagnosis')[num_cols].mean()
print(df_grouped)

# --- NORMALIZACIÓN DE VARIABLES NUMÉRICAS --- #
means = {}
stds = {}

def standardize(arr, col_name):
    mean = np.mean(arr)
    std = np.std(arr)
    means[col_name] = mean
    stds[col_name] = std
    return (arr - mean) / std

for col in num_cols:
    df[col] = standardize(df[col].values, col)

# Guardar los parámetros de normalización
norm_params = pd.DataFrame({'Variable': list(means.keys()), 'Mean': list(means.values()), 'Std': list(stds.values())})
norm_params.to_csv('normalization_params.csv', index=False)

# Guardar el dataset limpio
df.to_csv('dataset_limpio.csv', index=False)

print("Análisis completado y dataset limpio guardado. Parámetros de normalización almacenados.")









import numpy as np

def clean_data(df):
    # Replace out-of-range or negative values with NaN
    df['CellAdhesion'] = df['CellAdhesion'].apply(lambda x: x if 0 <= x <= 1 else np.nan)
    df['NuclearMembrane'] = df['NuclearMembrane'].apply(lambda x: x if 1 <= x <= 5 else np.nan)
    df['OxygenSaturation'] = df['OxygenSaturation'].apply(lambda x: x if 0 <= x <= 100 else np.nan)
    df['Vascularization'] = df['Vascularization'].apply(lambda x: x if 0 <= x <= 10 else np.nan)
    df['InflammationMarkers'] = df['InflammationMarkers'].apply(lambda x: x if 0 <= x <= 100 else np.nan)
    # Replace '???' in CellType with NaN
    df['CellType'] = df['CellType'].apply(lambda x: x if x != '???' else np.nan)
    # Replace negative values with NaN
    df = df.applymap(lambda x: np.nan if isinstance(x, (int, float)) and x < 0 else x)
    # Replace outliers with NaN using IQR method
    numeric_columns = [
        "CellSize", "CellShape", "NucleusDensity", "ChromatinTexture",
        "CytoplasmSize", "CellAdhesion", "MitosisRate", "NuclearMembrane",
        "GrowthFactor", "OxygenSaturation", "Vascularization", "InflammationMarkers"
    ]
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    # Identify outliers and replace them with NaN
    for column in numeric_columns:
        df[column] = df[column].mask(
            (df[column] < (Q1[column] - 1.5 * IQR[column])) | 
            (df[column] > (Q3[column] + 1.5 * IQR[column]))
        )
    # Add a column with NaN counts per row and sort by it
    df['NaN_Count'] = df.isna().sum(axis=1)  # Count NaN values per row
    df = df.sort_values(by='NaN_Count', ascending = False)  # Sort by NaN_Count column
    df = df[df['NaN_Count'] < 7]
    return df

df_clean = clean_data(df.copy())
df_clean.head()