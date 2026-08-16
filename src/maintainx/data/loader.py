from pathlib import Path

import pandas as pd

from maintainx.data import paths
from maintainx.data.errors import DataValidationError, MissingFilesError

SUBSETS: list[str] = ["FD001", "FD002", "FD003", "FD004"]

COLUMNS: list[str] = [
    "unit_id",
    "cycle",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

PROCESSED_COLUMNS: list[str] = COLUMNS + ["rul"]


def _validate_subset(dataset: str) -> None:
    if dataset not in SUBSETS:
        raise DataValidationError(
            f"Unknown C-MAPSS subset {dataset!r}; expected one of {', '.join(SUBSETS)}"
        )


def _resolve_data_dir(data_dir: Path | None) -> Path:
    resolved = Path(data_dir) if data_dir is not None else paths.CMAPSS_DIR
    if not resolved.is_dir():
        raise MissingFilesError(
            f"C-MAPSS data not found in '{resolved}'. "
            "Run `python -m maintainx.data.download` to extract the archive."
        )
    return resolved


def _resolve_path(kind: str, dataset: str, data_dir: Path | None) -> Path:
    _validate_subset(dataset)
    directory = _resolve_data_dir(data_dir)
    path = directory / f"{kind}_{dataset}.txt"
    if not path.is_file():
        raise MissingFilesError(
            f"Missing file: {path.name}. "
            "Run `python -m maintainx.data.download` to extract the archive."
        )
    return path


def _read_whitespace_table(path: Path, expected_columns: int) -> pd.DataFrame:
    try:
        table = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    except Exception as exc:
        raise DataValidationError(f"Could not parse {path.name}: {exc}") from exc
    table = table.dropna(axis=1, how="all")
    if table.shape[1] != expected_columns:
        raise DataValidationError(
            f"{path.name} has {table.shape[1]} columns; expected {expected_columns}"
        )
    return table


def load_train_subset(dataset: str = "FD001", data_dir: Path | None = None) -> pd.DataFrame:
    """Load a C-MAPSS training subset as a DataFrame with named columns."""
    path = _resolve_path("train", dataset, data_dir)
    table = _read_whitespace_table(path, len(COLUMNS))
    table.columns = COLUMNS
    return table


def load_test_subset(dataset: str = "FD001", data_dir: Path | None = None) -> pd.DataFrame:
    """Load a C-MAPSS test subset as a DataFrame with named columns."""
    path = _resolve_path("test", dataset, data_dir)
    table = _read_whitespace_table(path, len(COLUMNS))
    table.columns = COLUMNS
    return table


def load_rul_subset(dataset: str = "FD001", data_dir: Path | None = None) -> pd.Series:
    """Load the RUL file for a subset as a Series of remaining-useful-life values."""
    path = _resolve_path("RUL", dataset, data_dir)
    table = _read_whitespace_table(path, 1)
    return table.iloc[:, 0].rename("rul").astype("int64")
