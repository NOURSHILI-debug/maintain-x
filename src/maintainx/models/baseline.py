"""Baseline models for RUL prediction.

Simple models used to validate the feature engineering pipeline
before moving to more complex models (e.g. XGBoost).
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
def train_linear_regression(X_train, y_train):
    """Train a plain Linear Regression baseline.

    Args:
        X_train: Feature matrix.
        y_train: Target RUL values.

    Returns:
        Fitted LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, seed=42):
    """Train a Random Forest baseline.

    Args:
        X_train: Feature matrix.
        y_train: Target RUL values.
        n_estimators: Number of trees.
        max_depth: Max tree depth (kept shallow for a fast baseline).
        seed: Random seed for reproducibility.

    Returns:
        Fitted RandomForestRegressor model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, seed=42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=500, max_depth=5, learning_rate=0.01, subsample=0.7, colsample_bytree=0.7, seed=42):
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model