import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix, parallel_coordinates

def clean_data(df):
    df['CellAdhesion'] = df['CellAdhesion'].apply(lambda x: x if 0 <= x <= 1 else np.nan)
    df['NuclearMembrane'] = df['NuclearMembrane'].apply(lambda x: x if 1 <= x <= 5 else np.nan)
    df['OxygenSaturation'] = df['OxygenSaturation'].apply(lambda x: x if 0 <= x <= 100 else np.nan)
    df['Vascularization'] = df['Vascularization'].apply(lambda x: x if 0 <= x <= 10 else np.nan)
    df['InflammationMarkers'] = df['InflammationMarkers'].apply(lambda x: x if 0 <= x <= 100 else np.nan)
    df['CellType'] = df['CellType'].apply(lambda x: x if x != '???' else np.nan)
    df = df.applymap(lambda x: np.nan if isinstance(x, (int, float)) and x < 0 else x)
    numeric_columns = [
        "CellSize", "CellShape", "NucleusDensity", "ChromatinTexture",
        "CytoplasmSize", "CellAdhesion", "MitosisRate", "NuclearMembrane",
        "GrowthFactor", "OxygenSaturation", "Vascularization", "InflammationMarkers"
    ]
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    for column in numeric_columns:
        df[column] = df[column].mask(
            (df[column] < (Q1[column] - 1.5 * IQR[column])) | 
            (df[column] > (Q3[column] + 1.5 * IQR[column]))
        )
    df['NaN_Count'] = df.isna().sum(axis=1) 
    df = df.sort_values(by='NaN_Count', ascending = False)
    df = df[df['NaN_Count'] < 7]
    return df.drop(columns=['NaN_Count'])


def knn_value(base_row, df, target_col, feature_cols, k):
    df['distance'] = np.linalg.norm(df[feature_cols].values - base_row[feature_cols].values, axis=1)
    nearest_neighbors = df.nsmallest(k, 'distance')
    most_common_value = nearest_neighbors[target_col].dropna().mode()
    df.drop(columns=['distance'], inplace=True)
    return most_common_value[0] if not most_common_value.empty else np.nan

def calculate_correlation(df, feature1, feature2):
    return df[[feature1, feature2]].corr().iloc[0, 1]

def impute_missing_values(df, k):
    # Identificar columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for target_col in numeric_cols:
        if df[target_col].isna().sum() == 0:
            continue
        
        # Calcular correlaciones con la columna objetivo
        correlated_features = [col for col in numeric_cols if col != target_col and 
                               abs(calculate_correlation(df, col, target_col)) > 0.8]
        
        if not correlated_features:
            print(f"No hay características con correlación > 80% con {target_col}. Imputando con el promedio.")
            df[target_col].fillna(df[target_col].mean(), inplace=True)
            continue
        
        feature_cols = correlated_features
        
        for idx, row in df[df[target_col].isna()].iterrows():
            df.at[idx, target_col] = knn_value(row, df.dropna(subset=[target_col]), target_col, feature_cols, k)
    
    return df


df = pd.read_csv('Data/raw/cell_diagnosis_dev.csv')
df_clean = clean_data(df.copy())
df_clean_without_NaN = impute_missing_values(df_clean.copy(), 'CellType', k=5)