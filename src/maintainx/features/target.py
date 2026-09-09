import pandas as pd

from maintainx.features.config import RUL_CAP


def cap_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Cap the RUL target while preserving the rest of the data frame."""
    if cap < 0:
        raise ValueError("RUL cap must be non-negative")
    if "rul" not in df.columns:
        raise ValueError("data frame must contain a 'rul' column")
    result = df.copy()
    result["rul"] = result["rul"].clip(upper=cap)
    return result