
from typing import Tuple
import pandas as pd
import joblib
import logging
import numpy as np

def load_model(path: str):
    """
    Load a trained model/pipeline from disk. Fail fast if loading fails.
    """
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        # Fail loudly — callers should handle the exception and terminate the
        # process (e.g., the Streamlit app should stop early). Returning a
        # silent DummyModel masks failures and can lead to misleading demo
        # behavior.
        logging.exception(f"Failed to load model from '{path}': {e}")
        # Re-raise as FileNotFoundError to make the failure explicit to callers
        raise FileNotFoundError(f"Failed to load model from '{path}': {e}") from e


def load_trained_models(
    clf_path: str,
    reg_path: str,
):
    """ 
    Convenience function to load classification and regression models.
    """
    clf_model = load_model(clf_path)
    reg_model = load_model(reg_path)
    return clf_model, reg_model


def make_predictions(
    clf_model,
    reg_model,
    input_data: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Given:
      - trained classification model (Good_Investment)
      - trained regression model (Future_Price_5Y)
      - input dataframe with same feature columns as training

    This function enforces feature alignment by running the feature pipeline
    (deterministic transforms) and then validating that all expected features
    are present. It fails fast on any mismatch.

    Returns:
      - pred_label: 0/1 predicted Good_Investment
      - pred_proba: probability of Good_Investment = 1
      - pred_price: predicted Future_Price_5Y
    """
    from src.features.build_features import run_feature_pipeline, validate_features

    # Run deterministic feature pipeline to ensure engineered features are created
    input_processed = run_feature_pipeline(input_data.copy())

    # Enforce feature configuration alignment
    validate_features(input_processed, require_targets=False)

    # Ensure models can make predictions and fail early if not
    try:
        proba = clf_model.predict_proba(input_processed)[:, 1]
    except Exception as e:
        logging.exception(f"Classification model failed during predict_proba: {e}")
        raise

    try:
        price = reg_model.predict(input_processed)
    except Exception as e:
        logging.exception(f"Regression model failed during predict: {e}")
        raise

    label = (proba >= 0.5).astype(int)

    return pd.Series(label), pd.Series(proba), pd.Series(price)
