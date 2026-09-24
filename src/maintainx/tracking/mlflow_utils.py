"""Single entry point for MLflow configuration and run logging."""

import os
import subprocess

import mlflow
import numpy as np
from dotenv import load_dotenv

from maintainx.evaluation.metrics import evaluate_model

DEFAULT_TRACKING_URI = "http://localhost:5000"
RUL_CAP = 125


def _git_commit() -> str:
    """Short hash of the current commit, so every run is traceable to code."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def setup_mlflow(experiment_name: str) -> str:
    """Load .env, point MLflow at the server, select the experiment."""
    load_dotenv()  # fills os.environ from .env; does not override existing vars
    uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri


def tag_run() -> None:
    """Call inside `with mlflow.start_run():` to stamp the run with the git commit."""
    mlflow.set_tag("git_commit", _git_commit())


def last_cycle_mask(df, unit_col: str = "unit_id", cycle_col: str = "cycle") -> np.ndarray:
    """Return a boolean mask selecting the final observed cycle for each unit."""
    return np.asarray(df[cycle_col] == df.groupby(unit_col)[cycle_col].transform("max"))


def log_regression_run(run_name, model, params,
                       X_val, y_val, X_test, y_test, last_mask,
                       flavor="sklearn"):
    """Log params, val/test metrics (incl. last-cycle benchmark) and the model.

    Metric groups:
        val_*        validation split (held-out training engines)
        test_*       all test rows
        test_last_*  one prediction per test engine (FD001 benchmark protocol)
    """
    last_mask = np.asarray(last_mask)

    def predict(X):
        return np.clip(model.predict(X), 0, RUL_CAP)  # target is capped

    val_pred, test_pred = predict(X_val), predict(X_test)
    y_val, y_test = np.asarray(y_val), np.asarray(y_test)

    with mlflow.start_run(run_name=run_name):
        tag_run()
        mlflow.log_params({**params, "rul_cap": RUL_CAP, "pred_clip": f"0-{RUL_CAP}"})

        mlflow.log_metrics(
            {f"val_{k}": v for k, v in evaluate_model(y_val, val_pred).items()}
        )
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in evaluate_model(y_test, test_pred).items()}
        )
        mlflow.log_metrics(
            {f"test_last_{k}": v for k, v in evaluate_model(
                y_test[last_mask], test_pred[last_mask]).items()}
        )

        if flavor == "xgboost":
            mlflow.xgboost.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(model, name="model")
