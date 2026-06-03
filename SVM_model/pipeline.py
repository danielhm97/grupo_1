import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Resolve data path: prefer relative to this file, fallback to working directory
try:
    _BASE = Path(__file__).resolve().parents[1]
except NameError:
    _BASE = Path.cwd()
DATA_PATH = str(_BASE / 'data' / 'diabetes_prediction_dataset.csv')

SMOKING_MAP = {
    "No Info": "NoInfo",
    "never": "NeverSmoked",
    "ever": "HasSmoked",
    "not current": "HasSmoked",
    "former": "HasSmoked",
    "current": "Smoking",
}

TARGET = "diabetes"
CAT_FEATURES = ["gender", "smoking_history"]
NUM_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES + BINARY_FEATURES


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()
    df = df[df["bmi"] < 65]
    df = df[df["gender"] != "Other"]
    df["age"] = df["age"].round().astype(int)
    return df


def merge_smoking(df):
    df = df.copy()
    df["smoking_history"] = df["smoking_history"].map(SMOKING_MAP)
    return df


def preprocess(df, target=TARGET):
    df = clean_data(df)
    df = merge_smoking(df)
    X = df[ALL_FEATURES]
    y = df[target]
    return X, y


def build_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", sparse_output=False)
    binary_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_FEATURES),
            ("cat", categorical_transformer, CAT_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
        ]
    )
    return preprocessor


def build_svm_pipeline(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"):
    preprocessor = build_preprocessor()
    svm = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight,
              probability=True, random_state=42)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("svm", svm),
    ])
    return pipeline


def get_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
