# Heart Disease Prediction

Machine learning pipeline that predicts heart disease risk from patient health data and measures **how much data preprocessing improves model performance**.

## Overview

The project works with a dataset of **10,000 patient records** and **20 clinical and lifestyle features** (age, blood pressure, cholesterol, BMI, CRP level, smoking, stress level, sleep hours and more). Three classification models are trained twice, once on raw data and once after a full preprocessing pipeline, and the results are compared side by side.

## What it does

**Exploratory data analysis** (`eda.py`)
- Detects missing values and suspicious zero values in clinical measurements
- Correlation heatmap of every feature against the target
- Distribution plots by gender and stress level

**Preprocessing pipeline** (`preprocess.py`)
- Type conversion and missing-value imputation using the median
- Replacement of invalid negative values
- Outlier capping with the IQR method
- One-hot encoding of categorical features
- Feature engineering: age groups and cholesterol risk groups
- Standardization with `StandardScaler`
- Correlation-based feature selection (drops features with |r| < 0.1)

**Model training and comparison** (`train_models.py`)
- Logistic Regression, Decision Tree and Random Forest (balanced class weights)
- 80/20 train/test split with a fixed random seed for reproducibility
- Accuracy, confusion matrix, precision, recall and F1-score
- Bar chart comparing every model before and after preprocessing

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## Project structure

```
heart-disease-prediction/
├── data/
│   └── heart_disease.csv   # raw dataset (10,000 rows)
├── eda.py                  # exploratory analysis and plots
├── preprocess.py           # cleaning + feature engineering, writes data/heart_cleaned.csv
├── train_models.py         # trains and compares the three models
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
python eda.py            # explore the data
python preprocess.py     # build the cleaned dataset
python train_models.py   # train models and compare results
```

## Possible next steps

- Hyperparameter tuning with cross-validation (GridSearchCV)
- Handling class imbalance with SMOTE
- Adding gradient boosting models (XGBoost, LightGBM)
- Serving the best model through a small web API
