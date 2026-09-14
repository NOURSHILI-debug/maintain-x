"""Shared utilities for model training."""

import numpy as np


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