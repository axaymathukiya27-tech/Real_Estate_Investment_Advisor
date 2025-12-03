
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib


def split_data_for_modeling(
    df: pd.DataFrame,
    clf_target: str = "Good_Investment",
    reg_target: str = "Future_Price_5Y",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """
    Split the processed dataframe into train/test features and targets
    for both classification and regression.

    Returns:
        X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test,
        numeric_cols, categorical_cols
    """
    # Targets
    y_clf = df[clf_target]
    y_reg = df[reg_target]

    # Features (drop ID + targets)
    drop_cols = ["ID", clf_target, reg_target]
    X = df.drop(columns=drop_cols, errors="ignore")

    numeric_cols: List[str] = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols: List[str] = X.select_dtypes(exclude=[np.number]).columns.tolist()

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X,
        y_clf,
        y_reg,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf,
    )

    return (
        X_train,
        X_test,
        y_clf_train,
        y_clf_test,
        y_reg_train,
        y_reg_test,
        numeric_cols,
        categorical_cols,
    )


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - scales numeric columns
      - one-hot encodes categorical columns
    """
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def get_classification_models(
    preprocessor: ColumnTransformer,
) -> Dict[str, Pipeline]:
    """
    Create baseline classification models wrapped in pipelines.
    """
    models: Dict[str, Pipeline] = {}

    log_reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    models["logistic_regression"] = log_reg

    rf_clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    models["random_forest_classifier"] = rf_clf

    return models


def get_regression_models(
    preprocessor: ColumnTransformer,
) -> Dict[str, Pipeline]:
    """
    Create baseline regression models wrapped in pipelines.
    """
    models: Dict[str, Pipeline] = {}

    lin_reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    models["linear_regression"] = lin_reg

    rf_reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=80,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    models["random_forest_regressor"] = rf_reg

    return models


def save_model(model, path: str) -> None:
    """
    Save a trained model/pipeline to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
