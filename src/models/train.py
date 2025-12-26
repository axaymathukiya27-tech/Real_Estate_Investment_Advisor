
import os
from typing import Dict, Tuple, List
import json
from pathlib import Path

# Get absolute path to feature config
PROJECT_ROOT = Path(__file__).parent.parent.parent
FEATURE_CONFIG_PATH = PROJECT_ROOT / "src" / "features" / "feature_config.json"

import numpy as np
import pandas as pd
from src.data.load import load_raw_data


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import joblib
from .evaluate import classification_metrics, regression_metrics
from src.config import (
    MODEL_DIR,
    BEST_CLASSIFIER,
    BEST_REGRESSOR,
    RANDOM_STATE,
    TEST_SIZE,
)


def split_data_for_modeling(
    df: pd.DataFrame,
    clf_target: str = "Good_Investment",
    reg_target: str = "Future_Price_5Y",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
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

    # Load feature configuration
    with open(FEATURE_CONFIG_PATH, "r") as f:
        feature_config = json.load(f)

    numeric_cols: List[str] = feature_config["numeric_features"]
    categorical_cols: List[str] = feature_config["categorical_features"]

    # Keep only allowed features
    X = X[numeric_cols + categorical_cols]

    # We perform a single train/test split using StratifiedShuffleSplit on the
    # classification target (if possible) to maintain class balance while keeping
    # the same train/test indexes for both classification and regression tasks.
    # This prevents creating different test sets for the two tasks and avoids
    # stratifying the continuous regression target directly (which would be
    # inappropriate).
    from sklearn.model_selection import StratifiedShuffleSplit

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(sss.split(X, y_clf))

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)

        y_clf_train = y_clf.iloc[train_idx].reset_index(drop=True)
        y_clf_test = y_clf.iloc[test_idx].reset_index(drop=True)

        y_reg_train = y_reg.iloc[train_idx].reset_index(drop=True)
        y_reg_test = y_reg.iloc[test_idx].reset_index(drop=True)

    except Exception:
        # Fallback to a simple random split if stratification is not possible
        X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
            X,
            y_clf,
            y_reg,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
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
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    models["random_forest_classifier"] = rf_clf

    xgb_clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
            )),
        ]
    )
    models["xgboost_classifier"] = xgb_clf


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
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    models["random_forest_regressor"] = rf_reg

    xgb_reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=RANDOM_STATE,
            )),
        ]
    )
    models["xgboost_regressor"] = xgb_reg


    return models


def save_model(model, path: str) -> None:
    """
    Save a trained model/pipeline to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def train_baseline_models(
    df: pd.DataFrame,
    model_dir: str = MODEL_DIR,
    clf_target: str = "Good_Investment",
    reg_target: str = "Future_Price_5Y",
):
    """
    Train baseline classification and regression models.
    Saves best models to disk and returns metrics.
    """

    (
        X_train, X_test,
        y_clf_train, y_clf_test,
        y_reg_train, y_reg_test,
        numeric_cols, categorical_cols,
    ) = split_data_for_modeling(
        df,
        clf_target=clf_target,
        reg_target=reg_target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    reg_preprocessor = build_preprocessor(
        [c for c in numeric_cols if c != "Price_per_SqFt"],
        categorical_cols
    )

    clf_models = get_classification_models(preprocessor)
    reg_models = get_regression_models(reg_preprocessor)

    clf_metrics, reg_metrics = {}, {}

    import mlflow
    from .mlflow_setup import set_mlflow_experiment

    set_mlflow_experiment("real_estate_investment")
    
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="baseline_models",nested=True):

        for name, model in clf_models.items():
            model.fit(X_train, y_clf_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            metrics = classification_metrics(y_clf_test, preds, probs)
            clf_metrics[name] = metrics

            mlflow.log_metrics({f"clf_{name}_{k}": v for k, v in metrics.items()})

        for name, model in reg_models.items():
            model.fit(X_train, y_reg_train)
            preds = model.predict(X_test)

            metrics = regression_metrics(y_reg_test, preds)
            reg_metrics[name] = metrics

            mlflow.log_metrics({f"reg_{name}_{k}": v for k, v in metrics.items()})

        # ---- Save best models ----
        best_clf = max(clf_metrics, key=lambda k: clf_metrics[k]["f1"])
        best_reg = max(reg_metrics, key=lambda k: reg_metrics[k]["r2"])

        os.makedirs(model_dir, exist_ok=True)

        save_model(clf_models[best_clf], os.path.join(model_dir, BEST_CLASSIFIER))
        save_model(reg_models[best_reg], os.path.join(model_dir, BEST_REGRESSOR))

        mlflow.log_param("best_classifier", best_clf)
        mlflow.log_param("best_regressor", best_reg)

    return clf_models, reg_models, clf_metrics, reg_metrics
def main():
    """
    Entry point for training models from the command line.
    """
    print("Starting baseline model training...")

    # Load raw dataset
    df = load_raw_data("data/processed/housing_with_features.csv") 

    # Run baseline training
    clf_models, reg_models, clf_metrics, reg_metrics = train_baseline_models(df)

    print("Baseline training completed.")
    print("Classification metrics:", clf_metrics)
    print("Regression metrics:", reg_metrics)


if __name__ == "__main__":
    main()
