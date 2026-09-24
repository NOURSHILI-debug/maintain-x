"""Shared utilities for model training."""

from pathlib import Path

import numpy as np
import pandas as pd

from maintainx.features.pipeline import build_features

NON_FEATURE_COLS = ("unit_id", "cycle", "rul")


def _repo_root() -> Path:
    # src/maintainx/models/utils.py -> parents[3] is the repo root
    return Path(__file__).resolve().parents[3]


def _load_features():
    """Read the processed parquets and return (train_features, test_features)."""
    base_dir = _repo_root()
    train = pd.read_parquet(base_dir / "data/processed/FD001/train.parquet")
    test = pd.read_parquet(base_dir / "data/processed/FD001/test.parquet")
    return build_features(train, test)


def _feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def split_by_unit(df, val_fraction=0.2, seed=42):
    """Split a features DataFrame into train/val by engine unit_id.

    Splitting by whole engines (rather than random rows) avoids leaking
    information between train and validation, since rows from the same
    unit_id are correlated across cycles.
    """
    unit_ids = df["unit_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unit_ids)

    n_val = int(len(unit_ids) * val_fraction)
    val_units = set(unit_ids[:n_val])

    val_df = df[df["unit_id"].isin(val_units)]
    train_df = df[~df["unit_id"].isin(val_units)]
    return train_df, val_df


def load_train_val_split(val_fraction=0.2, seed=42):
    """Load processed data, build features, and split by unit_id.

    Returns:
        X_train, y_train, X_val, y_val
    """
    train_features, _ = _load_features()
    train_split, val_split = split_by_unit(train_features, val_fraction, seed)

    feature_cols = _feature_cols(train_split)
    X_train, y_train = train_split[feature_cols], train_split["rul"]
    X_val, y_val = val_split[feature_cols], val_split["rul"]
    return X_train, y_train, X_val, y_val


def load_test_set():
    """Load the test features with the same columns/order used for training.

    Returns:
        X_test:    test features (same feature columns as X_train).
        y_test:    capped true RUL for every test row.
        last_mask: boolean array, True on each engine's final cycle.
    """
    train_features, test_features = _load_features()

    missing = set(NON_FEATURE_COLS) - set(test_features.columns)
    assert not missing, (
        f"Test features lack {sorted(missing)}; the pipeline drops them."
    )

    feature_cols = _feature_cols(train_features)
    X_test = test_features[feature_cols]
    y_test = test_features["rul"]

    last_mask = (
        test_features.groupby("unit_id")["cycle"].transform("max")
        == test_features["cycle"]
    ).to_numpy()

    return X_test, y_test, last_mask