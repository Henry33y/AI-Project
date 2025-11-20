import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = os.path.join("data", "student_data.csv")


CURRENT_YEAR = 2025


TARGET_HIGH_GPA_THRESHOLD = 3.5
TARGET_OKAY_GPA_THRESHOLD = 2.5


BMI_BINS = [0, 18.5, 25, 30, np.inf]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese"]


@dataclass
class RegressionResult:
    model: Pipeline
    X_columns: list
    r2: float
    mse: float
    coefficients: pd.DataFrame


@dataclass
class ClassificationResult:
    model: Pipeline
    X_columns: list
    accuracy: float
    confusion: np.ndarray
    report: str
    coefficients: pd.DataFrame


def load_data(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = DATA_PATH
    df = pd.read_csv(path)

    required_cols = [
        "height",
        "weight",
        "BMI",
        "level",
        "faculty",
        "department",
        "yob",
        "GPA",
        "gender",
        "study_hours",
        "WASSCE_Aggregate",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["age"] = CURRENT_YEAR - df["yob"].astype(int)

    df["BMI_Category"] = pd.cut(df["BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False)

    df["High_GPA"] = (df["GPA"] >= TARGET_HIGH_GPA_THRESHOLD).astype(int)
    df["Okay_GPA"] = ((df["GPA"] >= TARGET_OKAY_GPA_THRESHOLD) & (df["GPA"] < TARGET_HIGH_GPA_THRESHOLD)).astype(int)

    return df


def _feature_spec(df: pd.DataFrame) -> Tuple[list, list]:
    numeric_features = [
        "WASSCE_Aggregate",
        "study_hours",
        "age",
        "BMI",
    ]
    categorical_features = [
        "faculty",
        "department",
        "level",
        "gender",
        "BMI_Category",
    ]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    return numeric_features, categorical_features


def _build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list]:
    numeric_features, categorical_features = _feature_spec(df)

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    all_features = numeric_features + categorical_features
    return preprocessor, all_features


def train_regression_model(df: pd.DataFrame) -> RegressionResult:
    df = df.dropna().copy()

    preprocessor, features = _build_preprocessor(df)

    X = df[features]
    y = df["GPA"].astype(float)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    regressor = model.named_steps["regressor"]
    pre = model.named_steps["preprocessor"]

    num_features, cat_features = _feature_spec(df)

    num_feature_names = num_features
    cat_feature_names = []
    if cat_features:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = list(ohe.get_feature_names_out(cat_features))

    feature_names = num_feature_names + cat_feature_names

    coef = regressor.coef_
    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
    }).sort_values(by="coefficient", ascending=False).reset_index(drop=True)

    return RegressionResult(model=model, X_columns=feature_names, r2=r2, mse=mse, coefficients=coefficients)


def train_high_gpa_classifier(df: pd.DataFrame) -> ClassificationResult:
    df = df.dropna().copy()

    preprocessor, features = _build_preprocessor(df)

    X = df[features]
    y = df["High_GPA"].astype(int)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X, y)
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    conf = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    classifier = model.named_steps["classifier"]
    pre = model.named_steps["preprocessor"]

    num_features, cat_features = _feature_spec(df)
    num_feature_names = num_features
    cat_feature_names = []
    if cat_features:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = list(ohe.get_feature_names_out(cat_features))

    feature_names = num_feature_names + cat_feature_names

    coef = classifier.coef_[0]
    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)

    return ClassificationResult(
        model=model,
        X_columns=feature_names,
        accuracy=acc,
        confusion=conf,
        report=report,
        coefficients=coefficients,
    )


def build_student_row_from_inputs(student_data: Dict[str, Any]) -> pd.DataFrame:
    """Construct a DataFrame for a single student with derived features.

    Expects keys compatible with the raw CSV schema: height, weight, BMI (optional),
    level, faculty, department, yob, GPA (placeholder ok), gender, study_hours,
    WASSCE_Aggregate.
    """

    df_input = pd.DataFrame([student_data])

    # Ensure BMI exists (compute from height/weight if missing or requested)
    if "BMI" not in df_input.columns or df_input["BMI"].isna().any():
        if {"height", "weight"}.issubset(df_input.columns):
            height_m = df_input["height"] / 100.0
            df_input["BMI"] = df_input["weight"] / (height_m ** 2)
        else:
            raise ValueError("BMI or (height, weight) must be provided for prediction.")

    # Derive age from yob
    if "age" not in df_input.columns and "yob" in df_input.columns:
        df_input["age"] = CURRENT_YEAR - df_input["yob"].astype(int)

    # Derive BMI category
    if "BMI_Category" not in df_input.columns:
        df_input["BMI_Category"] = pd.cut(df_input["BMI"], bins=BMI_BINS, labels=BMI_LABELS, right=False)

    return df_input


def predict_student(
    regression: RegressionResult,
    classifier: ClassificationResult,
    student_data: Dict[str, Any],
) -> Dict[str, Any]:
    df_input = build_student_row_from_inputs(student_data)

    gpa_pred = float(regression.model.predict(df_input)[0])
    high_prob = float(classifier.model.predict_proba(df_input)[0, 1])

    return {
        "predicted_gpa": gpa_pred,
        "high_gpa_probability": high_prob,
    }
