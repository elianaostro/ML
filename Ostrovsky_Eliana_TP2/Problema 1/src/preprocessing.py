import pandas as pd
import numpy as np

def clean_data(df):
    df['CellAdhesion'] = df['CellAdhesion'].apply(lambda x: x if 0 <= x <= 1 else np.nan)
    df['NuclearMembrane'] = df['NuclearMembrane'].apply(lambda x: x if 1 <= x <= 5 else np.nan)
    df['OxygenSaturation'] = df['OxygenSaturation'].apply(lambda x: x if 0 <= x <= 100 else np.nan)
    df['Vascularization'] = df['Vascularization'].apply(lambda x: x if 0 <= x <= 10 else np.nan)
    df['InflammationMarkers'] = df['InflammationMarkers'].apply(lambda x: x if 0 <= x <= 100 else np.nan)
    df['CellType'] = df['CellType'].apply(lambda x: x if x != '???' else np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
    numeric_columns = [
        "CellSize", "CellShape", "NucleusDensity", "ChromatinTexture",
        "CytoplasmSize", "CellAdhesion", "MitosisRate", "NuclearMembrane",
        "GrowthFactor", "OxygenSaturation", "Vascularization", "InflammationMarkers"
    ]
    Q1 = df[numeric_columns].quantile(0.1)
    Q3 = df[numeric_columns].quantile(0.9)
    IQR = Q3 - Q1
    for column in numeric_columns:
        df[column] = df[column].mask(
            (df[column] < (Q1[column] - 1.5 * IQR[column])) | 
            (df[column] > (Q3[column] + 1.5 * IQR[column]))
        )
    df['NaN_Count'] = df.isna().sum(axis=1) 
    df = df.sort_values(by='NaN_Count', ascending = False)
    df = df[df['NaN_Count'] < 7]

    df['Epthlial'] = (df['CellType'] == 'Epthlial').astype(int)
    df['Mesnchymal'] = (df['CellType'] == 'Mesnchymal').astype(int)
    df['GeneticMutation'] = (df['GeneticMutation'] == 'Presnt').astype(int)
    return df.drop(columns=['NaN_Count'])

def knn_value(base_row, df, target_col, feature_cols, k):
    """
    Calculate the k-nearest neighbors value for a given row in a DataFrame.

    Parameters:
    base_row (pd.Series): The row for which to calculate the k-nearest neighbors value.
    df (pd.DataFrame): The DataFrame containing the data.
    target_col (str): The name of the target column whose value is to be predicted.
    feature_cols (list of str): The list of feature column names to be used for distance calculation.
    k (int): The number of nearest neighbors to consider.

    Returns:
    The most common value among the k-nearest neighbors in the target column. 
    If there is no common value, returns NaN.
    """
    valid_features = [col for col in feature_cols if not pd.isna(base_row[col])]
    
    if not valid_features:
        return df[target_col].mean()  
    
    base_vector = np.array([base_row[col] for col in valid_features], dtype=float)
    distances = np.linalg.norm(df[valid_features].values - base_vector, axis=1)
    df = df.copy()
    df['distance'] = distances
    nearest_neighbors = df.nsmallest(k, 'distance')
    most_common_value = nearest_neighbors[target_col].dropna().mode()
    return most_common_value[0] if not most_common_value.empty else df[target_col].mean()

def calculate_correlation(df, feature1, feature2):
    return df[[feature1, feature2]].corr().iloc[0, 1]

def impute_missing_values(df, k):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for target_col in numeric_cols:
        if df[target_col].isna().sum() == 0:
            continue
        
        correlated_features = [col for col in numeric_cols if col != target_col and 
                               abs(calculate_correlation(df, col, target_col)) > 0.8]
        
        if not correlated_features:
            df[target_col] = df[target_col].fillna(df[target_col].mean())
            continue
        
        feature_cols = correlated_features
        
        for idx, row in df[df[target_col].isna()].iterrows():
            df.at[idx, target_col] = knn_value(row, df.dropna(subset=[target_col]), target_col, feature_cols, k)
    return df

def split_data(df, target_column, train_ratio=0.8, random_state=None):
    '''
    Split the DataFrame into training and validation sets.
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    '''
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    train_size = int(train_ratio * len(df_shuffled))
    
    X_train = df_shuffled.iloc[:train_size].drop(columns=[target_column])
    X_val = df_shuffled.iloc[train_size:].drop(columns=[target_column])
    
    y_train = df_shuffled.iloc[:train_size][target_column].values
    y_val = df_shuffled.iloc[train_size:][target_column].values
    
    return X_train, X_val, y_train, y_val
