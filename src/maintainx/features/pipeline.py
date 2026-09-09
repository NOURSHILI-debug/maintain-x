import pandas as pd

from maintainx.features.config import FLAT_SENSORS_FD001, ROLLING_WINDOW
from maintainx.features.rolling import add_rolling_features
from maintainx.features.selection import drop_constant_columns
from maintainx.features.target import cap_rul
from maintainx.features.temporal import add_delta_features


def build_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the same leakage-safe feature transformations to train and test."""
    sensor_columns = [column for column in train_df.columns if column.startswith("sensor_")]
    flat_sensors = {f"sensor_{sensor}" for sensor in FLAT_SENSORS_FD001}
    candidates = [
        column
        for column in sensor_columns
        + ["operational_setting_1", "operational_setting_2", "operational_setting_3"]
        if column in train_df.columns and column not in flat_sensors
    ]
    selected = drop_constant_columns(train_df, candidates)
    retained = [column for column in candidates if column in selected.columns]
    drop_columns = [column for column in candidates if column not in retained]
    train = train_df.drop(columns=drop_columns)
    test = test_df.drop(columns=[column for column in drop_columns if column in test_df.columns])
    train = add_delta_features(add_rolling_features(train, ROLLING_WINDOW))
    test = add_delta_features(add_rolling_features(test, ROLLING_WINDOW))
    if "rul" in train.columns:
        train = cap_rul(train)
    if "rul" in test.columns:
        test = cap_rul(test)
    return train, test