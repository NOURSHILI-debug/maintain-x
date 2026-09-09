import pandas as pd

from maintainx.features.config import ROLLING_WINDOW
from maintainx.features.errors import InvalidWindowError, LeakageError


def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add past-and-current rolling means for sensors within each engine."""
    if window < 1:
        raise InvalidWindowError("rolling window must be at least 1")
    if not {"unit_id", "cycle"}.issubset(df.columns):
        raise LeakageError("rolling features require unit_id and cycle")
    result = df.copy()
    ordered = result.sort_values(["unit_id", "cycle"], kind="stable")
    groups = ordered.groupby("unit_id", sort=False)
    for sensor in [column for column in df.columns if column.startswith("sensor_")]:
        values = groups[sensor].rolling(window, min_periods=1).mean()
        values = values.reset_index(level=0, drop=True)
        result.loc[ordered.index, f"{sensor}_rolling_mean"] = values.to_numpy()
    return result.loc[df.index]