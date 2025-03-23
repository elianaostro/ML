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
    
    return df

# Normalize data
def normalize(X, means=None, stds=None):
    X = X.copy()
    cols = ["area", "age", "rooms", "lat", "lon"]
    means = X[cols].mean() if means is None else means
    stds = X[cols].std() if stds is None else stds
    X[cols] = (X[cols] - means) / stds
    return X, means, stds

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
    
    def train_gradient_descent(self, lr=0.01, epochs=1000, clip_value=1e-2):
        m, n = self.X.shape
        self.coef = np.zeros((n, 1))
        
        for _ in range(epochs):
            gradients = (2/m) * self.X.T @ (self.X @ self.coef - self.y.reshape(-1, 1))
            gradients = np.clip(gradients, -clip_value, clip_value)

            if gradients.shape != self.coef.shape:
                gradients = gradients[:self.coef.shape[0], :]  # Ajustar el tamaño

            self.coef -= lr * gradients

    
    def predict(self, X_new):
        X_new = np.c_[np.ones((X_new.shape[0], 1)), X_new]
        return X_new @ self.coef
    
    def print_coefficients(self):
        print("Model Coefficients:")
        for name, coef in zip(self.feature_names, self.coef.flatten()):
            print(f"{name}: {coef:.2f}")

# Load and preprocess data
df = pd.read_csv("data/raw/casas_dev.csv")
df = convert_units(df)
df = impute_missing_values(df)

# Split and normalize data
X_train, X_val, y_train, y_val = split_data(df, "price")
X_train, train_mean, train_std = normalize(X_train)
X_val, _, _ = normalize(X_val, train_mean, train_std)

# Create and train the model
model_area = LinearRegression(X_train[['area']], y_train)

# print("Training with Pseudoinverse:")
# model_area.train_pseudoinverse()

# print_metrics(y_train, model_area.predict(X_train[['area']]), "Training")
# print_metrics(y_val, model_area.predict(X_val[['area']]), "Validation")
# model_area.print_coefficients()


# print("Training with Gradient Descent:")
model_area.train_gradient_descent(lr=0.1, epochs=1000, clip_value=1e-2)

print_metrics(y_train, model_area.predict(X_train[['area']]), "Training")
print_metrics(y_val, model_area.predict(X_val[['area']]), "Validation")
model_area.print_coefficients()