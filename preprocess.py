
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# leximi i te dhenave
df = pd.read_csv("data/heart_disease.csv")

print("=== DATA LOADED ===")
print(df.head())


# konvertimi ne numra ashtu qe te mund te lexohen
df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))


# check bazik
print("\n=== INFO ===")
print(df.info())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())


# selektimi i numrave te kolonave vetem
num_cols = df.select_dtypes(include=['int64', 'float64']).columns


# vlerat negative
print("\n=== NEGATIVE VALUES CHECK ===")
print((df[num_cols] < 0).sum())

for col in num_cols:
    df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
    df[col] = df[col].fillna(df[col].median())


# outliers
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

# handeling
cat_cols = ['cp', 'thal', 'slope', 'restecg']

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

#  FEATURE ENGINEERING

if 'age' in df.columns:
    df['age_group'] = pd.cut(df['age'],
                            bins=[0, 40, 55, 100],
                            labels=[1, 2, 3])

if 'chol' in df.columns:
    df['chol_group'] = pd.cut(df['chol'],
                             bins=[0, 200, 240, 600],
                             labels=[1, 2, 3])

# SCALING
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# CORRELATION & FEATURE SELECTION
print("\n=== FEATURE SELECTION ===")

target_col = None

# gjetja target automatikisht
for col in df.columns:
    if col.lower() in ['target', 'output', 'heartdisease']:
        target_col = col
        break

# fallback nese nuk gjendet
if target_col is None:
    target_col = df.columns[-1]

print("Target column:", target_col)

corr = df.corr(numeric_only=True)

low_corr_cols = corr[target_col][abs(corr[target_col]) < 0.1].index

# mos e fshij targetin
low_corr_cols = [col for col in low_corr_cols if col != target_col]

print("Dropping low correlation columns:", low_corr_cols)

df.drop(columns=low_corr_cols, inplace=True)

# final output
print("\n=== FINAL DATASET ===")
print(df.shape)
print(df.head())

# save clean data
df.to_csv("data/heart_cleaned.csv", index=False)

print("DONE! File saved as 'heart_cleaned.csv'")