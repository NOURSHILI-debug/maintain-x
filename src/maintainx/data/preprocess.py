import argparse
from pathlib import Path

import pandas as pd

from maintainx.data import loader, paths
from maintainx.data.errors import CMAPSSDataError, DataValidationError, MissingFilesError
from maintainx.data.validation import (
    REQUIRED_FILES,
    SUBSETS,
    validate_dataset_directory,
    validate_engine_trajectories,
    validate_rul,
    validate_schema,
)


def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'rul' column for training data: RUL = max cycle - current cycle per engine."""
    result = df.copy()
    max_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["rul"] = (max_cycle - result["cycle"]).astype("int64")
    return result


def prepare_test_rul(test_df: pd.DataFrame, rul_series: pd.Series) -> pd.DataFrame:
    """Align the provided RUL file with the final observed cycle of each test engine.

    The k-th RUL entry corresponds to the k-th engine appearing in the test file.
    For each engine: RUL(cycle) = provided_RUL + (max_cycle - cycle).
    """
    result = test_df.copy()
    rul_values = pd.Series(rul_series, dtype="int64").to_numpy()
    engines = list(result.groupby("unit_id", sort=False))
    if len(engines) != len(rul_values):
        raise DataValidationError(
            f"RUL file has {len(rul_values)} entries but test data has {len(engines)} engines"
        )
    parts: list[pd.DataFrame] = []
    for (unit_id, group), rul_end in zip(engines, rul_values):
        engine = group.copy()
        max_cycle = engine["cycle"].max()
        engine["rul"] = rul_end + (max_cycle - engine["cycle"])
        parts.append(engine)
    return pd.concat(parts, ignore_index=True)


def preprocess_dataset(
    dataset: str, raw_dir: Path | None = None, processed_dir: Path | None = None
) -> dict[str, Path]:
    """Load, label and validate a C-MAPSS subset, then save it as Parquet.

    Returns a mapping of artifact name to file path.
    """
    data_dir = Path(raw_dir) if raw_dir is not None else paths.CMAPSS_DIR
    output_dir = Path(processed_dir) if processed_dir is not None else paths.PROCESSED_DIR
    if dataset not in SUBSETS:
        raise DataValidationError(
            f"Unknown C-MAPSS subset {dataset!r}; expected one of {', '.join(SUBSETS)}"
        )
    present = validate_dataset_directory(data_dir)
    if dataset not in present:
        raise MissingFilesError(
            f"Subset {dataset} is incomplete in '{data_dir}'. "
            f"Required files: {', '.join(REQUIRED_FILES[dataset])}"
        )

    train = add_rul_labels(loader.load_train_subset(dataset, data_dir))
    test = prepare_test_rul(loader.load_test_subset(dataset, data_dir), loader.load_rul_subset(dataset, data_dir))

    validate_schema(train, columns=loader.PROCESSED_COLUMNS)
    validate_engine_trajectories(train)
    validate_rul(train)
    validate_schema(test, columns=loader.PROCESSED_COLUMNS)
    validate_engine_trajectories(test)
    validate_rul(test, expect_zero_final=False)

    out = output_dir / dataset
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.parquet"
    test_path = out / "test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    return {"train": train_path, "test": test_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess a C-MAPSS subset into data/processed.")
    parser.add_argument(
        "--dataset",
        default="FD001",
        choices=SUBSETS,
        help="C-MAPSS subset to preprocess (default: FD001).",
    )
    parser.add_argument("--raw-dir", default=str(paths.CMAPSS_DIR), help="Directory with extracted C-MAPSS files.")
    parser.add_argument("--processed-dir", default=str(paths.PROCESSED_DIR), help="Output directory for processed data.")
    args = parser.parse_args(argv)

    try:
        artifacts = preprocess_dataset(
            args.dataset,
            raw_dir=Path(args.raw_dir),
            processed_dir=Path(args.processed_dir),
        )
    except CMAPSSDataError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Processed subset {args.dataset}:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
