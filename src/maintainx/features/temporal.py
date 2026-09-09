import pandas as pd

from maintainx.features.errors import LeakageError


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-engine sensor deltas; first-cycle deltas are filled with zero."""
    if not {"unit_id", "cycle"}.issubset(df.columns):
        raise LeakageError("delta features require unit_id and cycle")
    result = df.copy()
    ordered = result.sort_values(["unit_id", "cycle"], kind="stable")
    groups = ordered.groupby("unit_id", sort=False)
    for sensor in [column for column in df.columns if column.startswith("sensor_")]:
        delta = groups[sensor].diff().fillna(0)
        result.loc[ordered.index, f"{sensor}_delta"] = delta.to_numpy()
    return result.loc[df.index]