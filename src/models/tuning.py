# src/models/tuning.py

"""
Model hyperparameter tuning utilities using RandomizedSearchCV + MLflow.

This module provides:
- run_random_search_with_mlflow: low-level helper for a single model
- tune_models_with_random_search: high-level function that:
    * samples the dataset
    * splits into train/test
    * builds the preprocessor
    * tunes RF + XGBoost for classification & regression
    * selects the best models
    * saves them to disk
    * logs the selected best models to MLflow for Model Registry
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier, XGBRegressor

import mlflow
import mlflow.sklearn

from .train import split_data_for_modeling, build_preprocessor
from .evaluate import classification_metrics, regression_metrics
from src.config import TEST_SIZE, RANDOM_STATE


def run_random_search_with_mlflow(
    model_name: str,
    estimator,
    param_distributions: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    task: str,        # "classification" or "regression"
    scoring: str,     # e.g. "f1" or "r2"
    n_iter: int = 20,
    cv=3,
    feature_count: int | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
    run_name: str | None = None,
):
    """
    Run RandomizedSearchCV for a given estimator and log everything to MLflow.

    Returns:
        best_estimator, best_params, test_metrics_dict
    """
    if run_name is None:
        run_name = f"tuning_{model_name}"

    from .mlflow_setup import set_mlflow_experiment, log_requirements

    # Ensure experiment name and tracking are standardized
    set_mlflow_experiment("real_estate_investment")

    with mlflow.start_run(run_name=run_name, nested=True):
        # Log environment artifact (best-effort)
        log_requirements()

        mlflow.set_tag("run_type", "tuning")
        mlflow.log_param("task", task)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_iter", n_iter)
        # log whether cv is an int or an object
        mlflow.log_param("cv_object", type(cv).__name__)
        mlflow.log_param("scoring", scoring)
        mlflow.log_param("n_rows_train", int(len(X_train)))
        mlflow.log_param("n_rows_test", int(len(X_test)))
        if feature_count is not None:
            mlflow.log_param("feature_count", int(feature_count))
        if test_size is not None:
            mlflow.log_param("test_size", float(test_size))
        if random_state is not None:
            mlflow.log_param("random_state", int(random_state))

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            verbose=1,
            n_jobs=-1,
            random_state=42,
        )

        search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_

        # Log best params and best CV score
        mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_score", float(best_cv_score))

        # Also log mean and std of CV metric at the best index
        try:
            best_idx = search.best_index_
            mean_test = float(search.cv_results_["mean_test_score"][best_idx])
            std_test = float(search.cv_results_["std_test_score"][best_idx])
            mlflow.log_metric("best_cv_mean", mean_test)
            mlflow.log_metric("best_cv_std", std_test)
        except Exception:
            # If cv_results_ unavailable / unexpected format, skip
            pass

        # Evaluate on test set
        y_pred = best_estimator.predict(X_test)

        if task == "classification":
            try:
                y_proba = best_estimator.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None
            metrics = classification_metrics(y_test, y_pred, y_proba)
        else:
            metrics = regression_metrics(y_test, y_pred)

        for m_name, m_value in metrics.items():
            mlflow.log_metric(f"test_{m_name}", float(m_value))

        # Log the tuned model itself as an artifact of this run
        mlflow.sklearn.log_model(best_estimator, artifact_path=f"{model_name}_tuned")

    return best_estimator, best_params, metrics


def tune_models_with_random_search(
    df: pd.DataFrame,
    sample_size: int = 50_000,
    n_iter: int = 15,
    cv: int = 3,
    models_dir: str | Path = "../models",
) -> Tuple[Dict, Dict]:
    """
    High-level function to tune all models:

    - Optionally samples df for speed
    - Splits into train/test for both tasks
    - Builds preprocessing pipeline
    - Tunes:
        * RandomForestClassifier
        * XGBClassifier
        * RandomForestRegressor
        * XGBRegressor
    - Selects best classifier by F1-score
    - Selects best regressor by R²
    - Saves them into models_dir as:
        * tuned_classification_model.pkl
        * tuned_regression_model.pkl
    - Logs the selected best models to MLflow in a dedicated run
      so they can be registered easily in the Model Registry.

    Returns:
        clf_results: {model_name: metrics_dict}
        reg_results: {model_name: metrics_dict}
    """
    # -------- Sample for speed --------
    if len(df) > sample_size:
        df_small = df.sample(sample_size, random_state=42)
    else:
        df_small = df.copy()

    # -------- Split + preprocessor --------
    (
        X_train,
        X_test,
        y_clf_train,
        y_clf_test,
        y_reg_train,
        y_reg_test,
        numeric_cols,
        categorical_cols,
    ) = split_data_for_modeling(df_small)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # For regression we intentionally **exclude** leakage-prone numeric features
    # such as `Price_per_SqFt` which encode price / size and can enable
    # trivial inversion when training regressors. This keeps regression honest.
    reg_numeric_cols = [c for c in numeric_cols if c not in ("Price_per_SqFt",)]
    reg_preprocessor = build_preprocessor(reg_numeric_cols, categorical_cols)

    # -------- Define base pipelines --------
    rf_clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    xgb_clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                ),
            ),
        ]
    )

    rf_reg = Pipeline(
        steps=[
            ("preprocess", reg_preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    xgb_reg = Pipeline(
        steps=[
            ("preprocess", reg_preprocessor),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=42,
                ),
            ),
        ]
    )

    # -------- Param distributions --------
    rf_clf_param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
    }

    xgb_clf_param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7, 9],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
    }

    rf_reg_param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
    }

    xgb_reg_param_dist = {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [3, 5, 7, 9],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
    }

    # -------- Run tuning – classification --------
    from sklearn.model_selection import StratifiedKFold, KFold

    clf_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    reg_cv = KFold(n_splits=cv, shuffle=True, random_state=42)

    rf_clf_best, rf_clf_best_params, rf_clf_metrics = run_random_search_with_mlflow(
        model_name="rf_classifier",
        estimator=rf_clf,
        param_distributions=rf_clf_param_dist,
        X_train=X_train,
        y_train=y_clf_train,
        X_test=X_test,
        y_test=y_clf_test,
        task="classification",
        scoring="f1",
        n_iter=n_iter,
        cv=clf_cv,
        feature_count=len(numeric_cols) + len(categorical_cols),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        run_name="tuning_rf_classifier",
    )

    xgb_clf_best, xgb_clf_best_params, xgb_clf_metrics = run_random_search_with_mlflow(
        model_name="xgb_classifier",
        estimator=xgb_clf,
        param_distributions=xgb_clf_param_dist,
        X_train=X_train,
        y_train=y_clf_train,
        X_test=X_test,
        y_test=y_clf_test,
        task="classification",
        scoring="f1",
        n_iter=n_iter,
        cv=clf_cv,
        feature_count=len(numeric_cols) + len(categorical_cols),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        run_name="tuning_xgb_classifier",
    )

    clf_results = {
        "rf_classifier": rf_clf_metrics,
        "xgb_classifier": xgb_clf_metrics,
    }

    # pick best by F1
    best_clf_name = max(clf_results, key=lambda k: clf_results[k]["f1"])
    best_clf_model = rf_clf_best if best_clf_name == "rf_classifier" else xgb_clf_best

    # -------- Run tuning – regression --------
    rf_reg_best, rf_reg_best_params, rf_reg_metrics = run_random_search_with_mlflow(
        model_name="rf_regressor",
        estimator=rf_reg,
        param_distributions=rf_reg_param_dist,
        X_train=X_train,
        y_train=y_reg_train,
        X_test=X_test,
        y_test=y_reg_test,
        task="regression",
        scoring="r2",
        n_iter=n_iter,
        cv=reg_cv,
        feature_count=len(numeric_cols) + len(categorical_cols),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        run_name="tuning_rf_regressor",
    )

    xgb_reg_best, xgb_reg_best_params, xgb_reg_metrics = run_random_search_with_mlflow(
        model_name="xgb_regressor",
        estimator=xgb_reg,
        param_distributions=xgb_reg_param_dist,
        X_train=X_train,
        y_train=y_reg_train,
        X_test=X_test,
        y_test=y_reg_test,
        task="regression",
        scoring="r2",
        n_iter=n_iter,
        cv=reg_cv,
        feature_count=len(numeric_cols) + len(categorical_cols),
        test_size=TEST_SIZE,
        run_name="tuning_xgb_regressor",
    )

    reg_results = {
        "rf_regressor": rf_reg_metrics,
        "xgb_regressor": xgb_reg_metrics,
    }

    # pick best by R²
    best_reg_name = max(reg_results, key=lambda k: reg_results[k]["r2"])
    best_reg_model = rf_reg_best if best_reg_name == "rf_regressor" else xgb_reg_best

    # -------- Save tuned models to disk --------
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    from src.config import BEST_CLASSIFIER, BEST_REGRESSOR

    tuned_clf_path = models_dir / BEST_CLASSIFIER
    tuned_reg_path = models_dir / BEST_REGRESSOR

    import joblib
    joblib.dump(best_clf_model, tuned_clf_path)
    joblib.dump(best_reg_model, tuned_reg_path)

    print("✔ Saved tuned classification model to:", tuned_clf_path)
    print("✔ Saved tuned regression model to    :", tuned_reg_path)

    # -------- ALSO log the selected best models to MLflow --------
    # This creates a single run that holds both final models,
    # which you can then register in the MLflow Model Registry.
    with mlflow.start_run(run_name="log_best_tuned_models", nested=True):
        mlflow.sklearn.log_model(
            sk_model=best_clf_model,
            artifact_path="best_clf_model",
        )
        mlflow.sklearn.log_model(
            sk_model=best_reg_model,
            artifact_path="best_reg_model",
        )
        active_run = mlflow.active_run()
        if active_run is not None:
            print(
                "📌 MLflow run_id for best models (use this for registry):",
                active_run.info.run_id,
            )

    return clf_results, reg_results
