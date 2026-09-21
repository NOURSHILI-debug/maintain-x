"""Shared utilities for model training."""

import numpy as np

import pandas as pd
from pathlib import Path
from maintainx.features.pipeline import build_features
def split_by_unit(df, val_fraction=0.2, seed=42):
    """Split a features DataFrame into train/val by engine unit_id.

    Splitting by whole engines (rather than random rows) avoids leaking
    information between train and validation, since rows from the same
    unit_id are correlated across cycles.

    Args:
        df: Features DataFrame, must contain a 'unit_id' column.
        val_fraction: Fraction of unique units to hold out for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df).
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

    Shared entry point so each model script doesn't duplicate this logic.
    """
    base_dir = Path(__file__).resolve().parents[3]  # repo root
    train = pd.read_parquet(base_dir / "data/processed/FD001/train.parquet")
    test = pd.read_parquet(base_dir / "data/processed/FD001/test.parquet")

    train_features, _ = build_features(train, test)
    train_split, val_split = split_by_unit(train_features, val_fraction, seed)

    feature_cols = [c for c in train_split.columns if c not in ("unit_id", "cycle", "rul")]
    X_train, y_train = train_split[feature_cols], train_split["rul"]
    X_val, y_val = val_split[feature_cols], val_split["rul"]
    return X_train, y_train, X_val, y_val
def load_test_set():
    """Test features aligned with the train feature columns, plus a last-cycle mask."""
    base_dir = Path(__file__).resolve().parents[3]
    train = pd.read_parquet(base_dir / "data/processed/FD001/train.parquet")
    test = pd.read_parquet(base_dir / "data/processed/FD001/test.parquet")

    train_features, test_features = build_features(train, test)
    feature_cols = [c for c in train_features.columns
                    if c not in ("unit_id", "cycle", "rul")]

    missing = {"unit_id", "cycle", "rul"} - set(test_features.columns)
    assert not missing, f"test features lack {missing}; the pipeline drops them"

    X_test = test_features[feature_cols]          # same columns/order as training
    y_test = test_features["rul"]
    last_mask = (
        test_features.groupby("unit_id")["cycle"].transform("max")
        == test_features["cycle"]
    ).to_numpy()
    return X_test, y_test, last_mask