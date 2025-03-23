import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Convert area units from sqft to m2
def convert_units(df):
    df = df.copy()
    df['area_units'] = df['area_units'].map({'sqft': 1, 'm2': 0})
    df['area_m2'] = np.where(df['area_units'], df['area'] * 0.092903, df['area'])
    return df

# Calculate KNN value
def knn_value(base_row, df, target_col, feature_cols, k):
    df['distance'] = np.linalg.norm(df[feature_cols].values - base_row[feature_cols].values, axis=1)
    nearest_neighbors = df.nsmallest(k, 'distance')
    most_common_value = nearest_neighbors[target_col].dropna().mode()
    df.drop(columns=['distance'], inplace=True)
    return most_common_value[0] if not most_common_value.empty else np.nan

# Assign number of rooms using KNN
def assign_rooms_knn(area, df, k=5):
    base_row = pd.DataFrame({'area': [area]})
    return knn_value(base_row.iloc[0], df, 'rooms', ['area'], k)

# Impute missing values for 'rooms' and 'age'
def impute_missing_values(df):
    df = df.copy()
    
    # Impute missing 'rooms' values
    mask = df['rooms'].isna()
    df.loc[mask, 'rooms'] = df.loc[mask, 'area'].apply(assign_rooms_knn, df=df)
    
    # Impute missing 'age' values
    house_mask = (df['is_house'] == 1) & (df['age'].isna())
    not_house_mask = (df['is_house'] == 0) & (df['age'].isna())
    df.loc[house_mask, 'age'] = df[df['is_house'] == 1]['age'].mean()
    df.loc[not_house_mask, 'age'] = df[df['is_house'] == 0]['age'].mean()

    # Impute other missing values using KNN
    for col in df.columns:
        if df[col].isna().any():
            mask = df[col].isna()
            df.loc[mask, col] = df.loc[mask].apply(lambda row: knn_value(row, df, col, df.columns.difference([col]), k=5), axis=1)
    
    return df

# Normalize data
def normalize(X, means=None, stds=None):
    X = X.copy()
    cols = X.select_dtypes(include=[np.number]).columns.difference(['is_house', 'has_pool', 'area_units', 'price'])
    means = X[cols].mean() if means is None else means
    stds = X[cols].std() if stds is None else stds
    X[cols] = (X[cols] - means) / stds
    return X, means, stds

# Preprocess data
def preprocess_data(df, train_mean=None, train_std=None):
    df = convert_units(df)
    df = impute_missing_values(df)
    df, _, _ = normalize(df, train_mean, train_std)
    return df

# Split data into training and validation sets
def split_data(df, target_column, train_ratio=0.8, random_state=None):
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_size = int(train_ratio * len(df_shuffled))
    X_train = df_shuffled.iloc[:train_size].drop(columns=[target_column])
    X_val = df_shuffled.iloc[train_size:].drop(columns=[target_column])
    y_train = df_shuffled.iloc[:train_size][target_column].values
    y_val = df_shuffled.iloc[train_size:][target_column].values
    return X_train, X_val, y_train, y_val

# Evaluate the model
def print_metrics(y_true, y_pred, label):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r_2 = 1 - mse / np.var(y_true)
    print(f"{label} Metrics:")
    print(f"- MSE: {mse:.2f}")
    print(f"- RMSE: {rmse:.2f}")
    print(f"- MAE: {mae:.2f}")
    print(f"- R^2: {r_2:.2f}\n")

# Linear Regression class
class LinearRegression:
    def __init__(self, X, y):
        self.X = np.c_[np.ones((X.shape[0], 1)), X]
        self.y = y
        self.coef = None
        self.feature_names = ['bias'] + list(X.columns)
    
    def train_pseudoinverse(self):
        self.coef = np.linalg.pinv(self.X) @ self.y
    
    def train_gradient_descent(self, lr=0.01, max_iters=1000, tol=1e-6):
        m, n = self.X.shape
        self.coef = np.zeros((n, 1))
        y = self.y.reshape(-1, 1)

        for _ in range(max_iters):
            gradients = (2/m) * self.X.T @ (self.X @ self.coef - y)
            if np.linalg.norm(gradients) < tol:
                break
            self.coef -= lr * gradients

    def predict(self, X_new):
        X_new = np.c_[np.ones((X_new.shape[0], 1)), X_new]
        return X_new @ self.coef
    
    def print_coefficients(self):
        print("Model Coefficients:")
        for name, coef in zip(self.feature_names, self.coef.flatten()):
            print(f"{name}: {coef:.2f}")
        print()

# Load and preprocess data
df = pd.read_csv("data/raw/casas_dev.csv")
df = convert_units(df)
df = impute_missing_values(df)

# Split and normalize data
X_train, X_val, y_train, y_val = split_data(df, "price")
X_train, train_mean, train_std = normalize(X_train)
X_train.describe()

cols = ['area_m2', 'is_house', 'has_pool', 'age', 'lat', 'lon','area_units']

# Create and train the model
model = LinearRegression(X_train[cols], y_train)

print("Training with Pseudoinverse:")
model.train_pseudoinverse()

print_metrics(y_train, model.predict(X_train[cols]), "Training")

X_val, _, _ = normalize(X_val, train_mean, train_std)
print_metrics(y_val, model.predict(X_val[cols]), "Validation")
model.print_coefficients()

# print("Training with Gradient Descent:")
# model.train_gradient_descent(lr=0.1, max_iters=1000, tol=1e-6)

# print_metrics(y_train, model.predict(X_train[cols]), "Training")
# print_metrics(y_val, model.predict(X_val[cols]), "Validation")
# model.print_coefficients()

df_amanda = pd.read_csv("data/raw/vivienda_Amanda.csv")
df_amanda = preprocess_data(df_amanda, train_mean, train_std)
y_amanda_pred = model.predict(df_amanda[cols])
print(f"Predicted price for Amanda's house: ${y_amanda_pred[0]:.2f}\n")

# Train the linear regression model
model = LinearRegression(X_train[cols], y_train)
model.train_pseudoinverse() 

# Calcular el valor por metro cuadrado
casas = df[df['is_house'] == 1]
valor_promedio_m2_casa = (casas['price'] / casas['area_m2']).mean()

print(f"El valor promedio por metro cuadrado de una casa es: {valor_promedio_m2_casa:.2f}\n")

# Impact of having a pool on the price
impact_pool = model.coef.flatten()[cols.index("has_pool") + 1]

print(f"Estimated impact of building a pool on the price: ${impact_pool:.2f}")