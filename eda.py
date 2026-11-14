import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('data/heart_disease.csv')

# Missing values
missing_values = df.isnull().sum()
print("Missing Values in Each Column:\n", missing_values) 

# Check suspicious zeros
columns_to_check = ['Blood Pressure', 'CRP Level', 'Triglyceride Level']
for col in columns_to_check:
    df[col] = pd.to_numeric(df[col], errors='coerce') 
    num_zeros = (df[col] == 0).sum()
    print(f"Column {col} has {num_zeros} suspicious zeros")

# Map target to numeric
df['Heart Disease Status'] = df['Heart Disease Status'].str.strip().map({'Yes':1, 'No':0})
#str.strip  per heqjen e hapsirave 
# Correlation
numeric_cols = df.select_dtypes(include='number').columns
corr_matrix = df[numeric_cols].corr()
target_corr = corr_matrix[['Heart Disease Status']].sort_values(by='Heart Disease Status', ascending=False)
#rendit nga me i madhi 

# Heatmap
plt.figure(figsize=(6,8))
sns.heatmap(target_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1) 
#hartemengjyra, shfaq numrat brenda kutive , kufinjet e shkalles
plt.title("Correlation with Heart Disease Status")
plt.show()

categorical_cols = ['Gender', 'Stress Level'] 
for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue='Heart Disease Status', data=df)
    plt.title(f"{col} count by Heart Disease Status")
    plt.show()