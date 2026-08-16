from pathlib import Path

import pandas as pd

from maintainx.data.errors import DataValidationError, MissingFilesError
from maintainx.data.loader import COLUMNS, SUBSETS

REQUIRED_FILES: dict[str, list[str]] = {
    subset: [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"] for subset in SUBSETS
}


def validate_dataset_directory(data_dir: Path) -> set[str]:
    """Check that the required FD001 files exist and detect all complete subsets.

    Returns the set of complete subsets found in the directory.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise MissingFilesError(f"C-MAPSS data directory not found: {data_dir}")
    missing = [name for name in REQUIRED_FILES["FD001"] if not (data_dir / name).is_file()]
    if missing:
        raise MissingFilesError(
            "Missing required C-MAPSS files:\n" + "\n".join(f"- {name}" for name in missing)
        )
    return {
        subset for subset in SUBSETS if all((data_dir / name).is_file() for name in REQUIRED_FILES[subset])
    }


def validate_schema(
    df: pd.DataFrame, columns: list[str] = COLUMNS, required_fields: tuple[str, ...] = ("unit_id", "cycle")
) -> None:
    """Check column names, count and required-field NaN presence."""
    if list(df.columns) != list(columns):
        raise DataValidationError(
            f"Unexpected columns {list(df.columns)}; expected {list(columns)}"
        )
    missing = [column for column in required_fields if df[column].isna().any()]
    if missing:
        raise DataValidationError(f"Unexpected NaN values in required fields: {missing}")


def validate_engine_trajectories(df: pd.DataFrame) -> None:
    """Check that unit ids and cycles are positive and cycles strictly increase per engine."""
    if not bool((df["unit_id"] > 0).all()):
        raise DataValidationError("unit_id must be positive")
    if not bool((df["cycle"] > 0).all()):
        raise DataValidationError("cycle must be positive")
    ordered = df.sort_values(["unit_id", "cycle"])
    steps = ordered.groupby("unit_id", sort=False)["cycle"].diff().dropna()
    if not bool((steps > 0).all()):
        raise DataValidationError("cycles must strictly increase within each engine")


def validate_rul(df: pd.DataFrame, expect_zero_final: bool = True) -> None:
    """Check RUL values are non-negative and consistent with the cycle trajectory."""
    if "rul" not in df.columns:
        raise DataValidationError("data frame is missing the 'rul' column")
    if not bool((df["rul"] >= 0).all()):
        raise DataValidationError("RUL must be non-negative")
    if expect_zero_final:
        max_cycle = df.groupby("unit_id")["cycle"].transform("max")
        if not bool((df.loc[df["cycle"] == max_cycle, "rul"] == 0).all()):
            raise DataValidationError("RUL must be 0 at the final observed cycle")
    ordered = df.sort_values(["unit_id", "cycle"])
    steps = ordered.groupby("unit_id", sort=False)["rul"].diff().dropna()
    if not bool((steps <= 0).all()):
        raise DataValidationError("RUL must not increase as cycle increases within an engine")
