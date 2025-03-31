import pandas as pd
import numpy as np

def convert_units(df):
    """
    Convert area units from square feet to square meters in a DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'area' and 'area_units' columns.
                           'area_units' should have values 'sqft' for square feet and 'm2' for square meters.

    Returns:
    pandas.DataFrame: A new DataFrame with an additional 'area_m2' column where the area is converted to square meters.
    """
    df = df.copy()
    df['area_units'] = df['area_units'].map({'sqft': 1, 'm2': 0})
    df['area_m2'] = np.where(df['area_units'], df['area'] * 0.092903, df['area'])
    return df

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
    df['distance'] = np.linalg.norm(df[feature_cols].values - base_row[feature_cols].values, axis=1)
    nearest_neighbors = df.nsmallest(k, 'distance')
    most_common_value = nearest_neighbors[target_col].dropna().mode()
    df.drop(columns=['distance'], inplace=True)
    return most_common_value[0] if not most_common_value.empty else np.nan

def assign_rooms_knn(area, df, k=5):
    """
    Assigns the number of rooms to a given area using the k-Nearest Neighbors (k-NN) algorithm.
    Parameters:
    area (float): The area for which the number of rooms needs to be assigned.
    df (pandas.DataFrame): The DataFrame containing the dataset with 'area' and 'rooms' columns.
    k (int, optional): The number of nearest neighbors to consider. Default is 5.
    Returns:
    int: The estimated number of rooms for the given area.
    """

    base_row = pd.DataFrame({'area': [area]})
    return knn_value(base_row.iloc[0], df, 'rooms', ['area'], k)

def impute_missing_values(df):
    """
    Impute missing values in the DataFrame.
    This function imputes missing values in the DataFrame `df` using different strategies:
    1. For the 'rooms' column, it uses a K-Nearest Neighbors (KNN) approach based on the 'area' column.
    2. For the 'age' column, it imputes missing values based on the mean age of houses and non-houses separately.
    3. For all other columns with missing values, it uses a KNN approach considering all other columns.
    Parameters:
    df (pd.DataFrame): The input DataFrame with potential missing values.
    Returns:
    pd.DataFrame: The DataFrame with imputed missing values.
    """

    mask = df['rooms'].isna()
    df.loc[mask, 'rooms'] = df.loc[mask, 'area'].apply(assign_rooms_knn, df=df)
    
    house_mask = (df['is_house'] == 1) & (df['age'].isna())
    not_house_mask = (df['is_house'] == 0) & (df['age'].isna())
    df.loc[house_mask, 'age'] = df[df['is_house'] == 1]['age'].mean()
    df.loc[not_house_mask, 'age'] = df[df['is_house'] == 0]['age'].mean()

    for col in df.columns:
        if df[col].isna().any():
            mask = df[col].isna()
            df.loc[mask, col] = df.loc[mask].apply(lambda row: knn_value(row, df, col, df.columns.difference([col]), k=5), axis=1)
    
    return df


def normalize(X, means=None, stds=None):
    """
    Normalizes numeric columns in the dataset.
    
    Args:
        X: DataFrame to normalize
        means: Optional means to use for normalization
        stds: Optional standard deviations to use for normalization
        
    Returns:
        Normalized DataFrame and the means and stds used
    """
    X = X.copy()
    cols = X.select_dtypes(include=[np.number]).columns.difference(['is_house', 'has_pool', 'area_units', 'price'])    
    X[cols] = X[cols].astype(float)
    means = X[cols].mean() if means is None else means
    stds = X[cols].std() if stds is None else stds
    X.loc[:, cols] = (X[cols] - means) / stds
    return X, means, stds

def preprocess_data(df):
    """
    Preprocess the input DataFrame by converting units and imputing missing values.
    Parameters:
    df (pandas.DataFrame): The input DataFrame to preprocess.
    Returns:
    pandas.DataFrame: The preprocessed DataFrame with units converted and missing values imputed.
    """
    df = convert_units(df)
    df = impute_missing_values(df)
    return df

def add_relative_location(df):
    '''
    Add relative location features to the DataFrame.
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    Returns:
    pandas.DataFrame: The DataFrame with added relative location features.
    '''
    lat_mean_1 = df.loc[df['area_units'] == 1, 'lat'].mean()
    lat_mean_0 = df.loc[df['area_units'] == 0, 'lat'].mean()
    lon_mean_1 = df.loc[df['area_units'] == 1, 'lon'].mean()
    lon_mean_0 = df.loc[df['area_units'] == 0, 'lon'].mean()

    df['dist_center'] = np.where(
        df['area_units'] == 1,
        np.sqrt((df['lat'] - lat_mean_1)**2 + (df['lon'] - lon_mean_1)**2),
        np.sqrt((df['lat'] - lat_mean_0)**2 + (df['lon'] - lon_mean_0)**2)
    )
    return df

def area_x_price(df):
    """
    Adds two new columns 'area_1' and 'area_0' to the DataFrame based on the 'area_units' column.
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing 'area_units' and 'area_m2' columns.
    Returns:
    pandas.DataFrame: The DataFrame with added 'area_1' and 'area_0' columns.
    """
    df['area_1'] = 0.0
    df['area_0'] = 0.0

    df['area_1'] = np.where(df['area_units'] == 1, df['area_m2'], 0)
    df['area_0'] = np.where(df['area_units'] == 0, df['area_m2'], 0)
    return df

def add_squared_features(df):
    '''
    Add squared features for 'age' and 'area' to the DataFrame.
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    Returns:
    pandas.DataFrame: The DataFrame with added squared features.
    '''
    df['age_squared'] = df['age'] ** 2
    df['area_squared'] = df['area'] ** 2
    return df


def add_features(df):
    '''
    Add relative location and area x price features to the DataFrame.
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    Returns:
    pandas.DataFrame: The DataFrame with added features.
    '''
    df = add_relative_location(df)
    df = area_x_price(df)
    df = add_squared_features(df)
    return df

def generate_polynomial_features(df, total_new_features = 300):
    '''
    Generate new polynomial features for the numerical features in the DataFrame.
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    total_new_features (int): The total number of new polynomial features to generate.
    Returns:
    pandas.DataFrame: The DataFrame with added polynomial features.
    '''

    numerical_features = df.select_dtypes(include=[np.number]).columns.difference(['price']).tolist()
    new_features = {}
    powers = list(range(2, 12))  
    
    num_features = len(numerical_features)
    powers_per_feature = total_new_features // num_features
    extra_powers = total_new_features % num_features
    
    for i, feature in enumerate(numerical_features):
        num_powers = powers_per_feature + (1 if i < extra_powers else 0)
        selected_powers = np.random.choice(powers, size=num_powers, replace=True)
        
        for p in selected_powers:
            new_feature_name = f"{feature}_pow{p}"
            new_features[new_feature_name] = df[feature] ** p
    
    df_expanded = df.copy()
    df_expanded = pd.concat([df_expanded, pd.DataFrame(new_features)], axis=1)
    
    return df_expanded

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


def split_and_normalize(df):
    '''
    Split the DataFrame into training and validation sets and normalize the features.
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    Returns:
    X_train (pandas.DataFrame): The normalized training features.
    X_val (pandas.DataFrame): The normalized validation features.
    y_train (np.ndarray): The training target values.
    y_val (np.ndarray): The validation target values.
    '''
    X_train, X_val, y_train, y_val = split_data(df, "price")
    X_train, train_mean, train_std = normalize(X_train)
    X_val, _, _ = normalize(X_val, train_mean, train_std)
    return X_train, X_val, y_train, y_val