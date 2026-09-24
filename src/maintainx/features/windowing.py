"""Sliding-window sequence construction for the RUL transformer."""

import numpy as np

WINDOW = 30


def _feature_cols(df):
    return [c for c in df.columns if c not in ("unit_id", "cycle", "rul")]


def make_train_windows(df, window=WINDOW, stride=5):
    """Slide a window over every engine's full history.

    Each engine contributes multiple overlapping windows, one per stride
    step, labeled with the RUL at the window's LAST cycle.

    Returns:
        X: (N, window, n_features) float32
        y: (N,) float32, RUL at each window's last cycle
    """
    feature_cols = _feature_cols(df)
    X_list, y_list = [], []

    for _, unit_df in df.groupby("unit_id"):
        unit_df = unit_df.sort_values("cycle")
        values = unit_df[feature_cols].to_numpy(dtype=np.float32)
        ruls = unit_df["rul"].to_numpy(dtype=np.float32)

        n = len(unit_df)
        if n < window:
            continue

        for start in range(0, n - window + 1, stride):
            end = start + window
            X_list.append(values[start:end])
            y_list.append(ruls[end - 1])

    return np.stack(X_list), np.array(y_list, dtype=np.float32)


def make_last_window(df, window=WINDOW):
    """One window per engine: its LAST `window` cycles (benchmark protocol).

    Returns:
        X: (n_engines, window, n_features) float32
        y: (n_engines,) float32, RUL at each engine's last cycle
        unit_ids: (n_engines,) the engine each row corresponds to
    """
    feature_cols = _feature_cols(df)
    X_list, y_list, unit_list = [], [], []

    for unit_id, unit_df in df.groupby("unit_id"):
        unit_df = unit_df.sort_values("cycle")
        values = unit_df[feature_cols].to_numpy(dtype=np.float32)
        ruls = unit_df["rul"].to_numpy(dtype=np.float32)

        n = len(unit_df)
        assert n >= window, f"unit {unit_id} has only {n} cycles, < window={window}"

        X_list.append(values[-window:])
        y_list.append(ruls[-1])
        unit_list.append(unit_id)

    return (np.stack(X_list), np.array(y_list, dtype=np.float32),
            np.array(unit_list))