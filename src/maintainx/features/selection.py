from collections.abc import Sequence

import pandas as pd

from maintainx.features.errors import FeatureSelectionError


def drop_constant_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Drop requested columns with zero variance or a single unique value."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise FeatureSelectionError(f"Missing columns: {missing}")
    constant = []
    for column in columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            if series.var(ddof=0) == 0:
                constant.append(column)
        elif series.nunique(dropna=False) <= 1:
            constant.append(column)
    return df.drop(columns=constant).copy()