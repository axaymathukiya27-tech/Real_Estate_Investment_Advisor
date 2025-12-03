
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def regression_metrics(
    y_true,
    y_pred,
) -> Dict[str, float]:
    """
    Compute standard regression metrics.

    NOTE: We compute RMSE manually from MSE instead of using
    mean_squared_error(..., squared=False) so it works with
    any sklearn version.
    """
    mae = mean_absolute_error(y_true, y_pred)

    # Old sklearn versions don't support squared=...
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

    return metrics
