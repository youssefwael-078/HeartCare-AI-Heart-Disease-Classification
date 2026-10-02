"""
preprocessing.py
----------------
Handles all data loading, cleaning, and preparation steps
for the Heart Disease Classification project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "dataset", "heart.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# ── Feature metadata (used in UI explanations) ───────────────────────────────
FEATURE_INFO = {
    "age":      {"label": "Age",               "desc": "Age of the patient in years",                          "type": "number"},
    "sex":      {"label": "Sex",               "desc": "1 = Male, 0 = Female",                                 "type": "number"},
    "cp":       {"label": "Chest Pain Type",   "desc": "0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic", "type": "number"},
    "trestbps": {"label": "Resting BP",        "desc": "Resting blood pressure (mm Hg)",                       "type": "number"},
    "chol":     {"label": "Cholesterol",       "desc": "Serum cholesterol (mg/dl)",                            "type": "number"},
    "fbs":      {"label": "Fasting Blood Sugar","desc": "Fasting blood sugar > 120 mg/dl (1=True, 0=False)",   "type": "number"},
    "restecg":  {"label": "Resting ECG",       "desc": "0=Normal, 1=ST-T wave abnormality, 2=LV hypertrophy", "type": "number"},
    "thalach":  {"label": "Max Heart Rate",    "desc": "Maximum heart rate achieved",                          "type": "number"},
    "exang":    {"label": "Exercise Angina",   "desc": "Exercise induced angina (1=Yes, 0=No)",                "type": "number"},
    "oldpeak":  {"label": "ST Depression",     "desc": "ST depression induced by exercise relative to rest",   "type": "float"},
    "slope":    {"label": "Slope",             "desc": "Slope of peak exercise ST segment (0,1,2)",            "type": "number"},
    "ca":       {"label": "Major Vessels",     "desc": "Number of major vessels colored by fluoroscopy (0-3)", "type": "number"},
    "thal":     {"label": "Thalassemia",       "desc": "1=Normal, 2=Fixed defect, 3=Reversible defect",        "type": "number"},
}

TARGET_CLASSES = {0: "No Disease (Healthy)", 1: "Heart Disease Present"}


def load_data():
    """Load the raw CSV dataset and return a DataFrame."""
    df = pd.read_csv(DATA_PATH)
    return df


def get_dataset_summary(df):
    """Return a dictionary of key dataset statistics for the UI."""
    summary = {
        "rows":          int(df.shape[0]),
        "cols":          int(df.shape[1]),
        "missing":       int(df.isnull().sum().sum()),
        "duplicates":    int(df.duplicated().sum()),
        "class_counts":  df["target"].value_counts().to_dict(),
        "feature_names": list(df.drop("target", axis=1).columns),
        "dtypes":        df.dtypes.astype(str).to_dict(),
        "describe":      df.describe().round(2).to_dict(),
    }
    return summary


def preprocess(df):
    """
    Full preprocessing pipeline:
      1. Drop duplicates
      2. Handle missing values (median imputation)
      3. Cap outliers using IQR method
      4. Split features / target
      5. Scale features with StandardScaler
      6. Train / test split (80/20, stratified)
    Returns X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # 1. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 2. Impute missing values with column median (robust to skew)
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # 3. Outlier capping (IQR) — applied only to continuous features
    continuous = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    for col in continuous:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    # 4. Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]
    feature_names = list(X.columns)

 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save the scaler for use during prediction
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    df = load_data()
    print("Dataset loaded:", df.shape)
    print(get_dataset_summary(df))
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
    print("Train size:", X_train.shape, "| Test size:", X_test.shape)
