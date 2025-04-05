import pandas as pd
from src import metrics, models, preprocessing

df = pd.read_csv('Ostrovsky_Eliana_TP2/Problema 1/Data/raw/cell_diagnosis_dev.csv')
df_clean = preprocessing.clean_data(df.copy())
df_clean_without_NaN = preprocessing.impute_missing_values(df_clean.copy(), k=5)
# df_clean_without_NaN.info()

# df_clean_without_NaN.hist(figsize=(15, 10))
# plt.tight_layout()
# plt.show()

df_clean_without_NaN = df_clean_without_NaN.drop(columns=["CellType"])
X_train, X_val, y_train, y_val = preprocessing.split_data(df_clean_without_NaN, "Diagnosis")
model = models.LogisticRegression(X_train, y_train)
model.train()
metrics.print_classification_metrics(y_train, model.predict(X_train), model.predict_proba(X_train), "Training")
metrics.print_classification_metrics(y_val, model.predict(X_val), model.predict_proba(X_val), "Validation")
