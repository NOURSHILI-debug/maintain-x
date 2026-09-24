"""Evaluation metrics for RUL prediction models.

Includes standard regression metrics (RMSE, MAE) and the NASA C-MAPSS
asymmetric scoring function used in published benchmarks.
"""

import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


def compute_rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return root_mean_squared_error(y_true, y_pred)


def compute_mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def compute_nasa_score(y_true, y_pred):
    """NASA C-MAPSS asymmetric scoring function.

    Penalizes late predictions (underestimating RUL, i.e. predicting
    the engine has more life left than it actually does) more heavily
    than early predictions, reflecting real maintenance risk: predicting
    failure too late is dangerous, predicting it too early just costs
    some engine life.

    For d = predicted_rul - actual_rul:
        d < 0 (early prediction): score = exp(-d/13) - 1
        d >= 0 (late prediction): score = exp(d/10) - 1

    Lower is better. Score is summed across all samples.

    Args:
        y_true: Array-like of actual RUL values.
        y_pred: Array-like of predicted RUL values.

    Returns:
        float: total NASA score across all samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    d = y_pred - y_true

    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score))


def evaluate_model(y_true, y_pred):
    """Compute all metrics for a set of predictions.

    Args:
        y_true: Array-like of actual RUL values.
        y_pred: Array-like of predicted RUL values.

    Returns:
        dict with keys 'rmse', 'mae', 'nasa_score'.
    """
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "nasa_score": compute_nasa_score(y_true, y_pred),
    }