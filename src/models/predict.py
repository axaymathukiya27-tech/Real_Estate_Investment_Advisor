
from typing import Tuple
import pandas as pd
import joblib


def load_model(path: str):
    """
    Load a trained model/pipeline from disk.
    """
    return joblib.load(path)


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

    Returns:
      - pred_label: 0/1 predicted Good_Investment
      - pred_proba: probability of Good_Investment = 1
      - pred_price: predicted Future_Price_5Y
    """
    # Classification
    proba = clf_model.predict_proba(input_data)[:, 1]
    label = (proba >= 0.5).astype(int)

    # Regression
    price = reg_model.predict(input_data)

    return pd.Series(label), pd.Series(proba), pd.Series(price)
