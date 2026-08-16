import zipfile

import pandas as pd
import pytest

from maintainx.data import loader
from maintainx.data.download import (
    extract_dataset,
    find_archive,
    is_cmapss_archive,
    validate_archive,
)
from maintainx.data.errors import (
    ArchiveNotFoundError,
    DataValidationError,
    InvalidArchiveError,
    MissingFilesError,
)
from maintainx.data.preprocess import add_rul_labels, prepare_test_rul, preprocess_dataset
from maintainx.data.validation import (
    validate_dataset_directory,
    validate_engine_trajectories,
    validate_rul,
    validate_schema,
)


def _sensor_row(unit_id: int, cycle: int) -> list[int | float]:
    return [unit_id, cycle, 0.1, 0.2, 0.3] + [float(unit_id + cycle) for _ in range(1, 22)]


def _write_subset(directory, subset: str = "FD001") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    train_cycles = {1: 5, 2: 4}
    test_cycles = {1: 3, 2: 2}
    train_lines = [
        " ".join(str(v) for v in _sensor_row(unit_id, cycle)) + "   "
        for unit_id, n_cycles in train_cycles.items()
        for cycle in range(1, n_cycles + 1)
    ]
    test_lines = [
        " ".join(str(v) for v in _sensor_row(unit_id, cycle)) + "   "
        for unit_id, n_cycles in test_cycles.items()
        for cycle in range(1, n_cycles + 1)
    ]
    (directory / f"train_{subset}.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (directory / f"test_{subset}.txt").write_text("\n".join(test_lines) + "\n", encoding="utf-8")
    (directory / f"RUL_{subset}.txt").write_text("2\n1\n", encoding="utf-8")


def _make_archive(tmp_path, layout: str = "direct", name: str = "CMAPSSData.zip"):
    src = tmp_path / "src"
    _write_subset(src)
    archive = tmp_path / name
    with zipfile.ZipFile(archive, "w") as zf:
        for file in src.iterdir():
            arcname = f"CMAPSSData/{file.name}" if layout == "nested" else file.name
            zf.write(file, arcname)
    return archive


# --- archive validation ------------------------------------------------------


def test_columns_length():
    assert len(loader.COLUMNS) == 26


def test_validate_archive_accepts_valid_zip(tmp_path):
    archive = _make_archive(tmp_path)
    validate_archive(archive)


def test_validate_archive_rejects_non_zip(tmp_path):
    archive = tmp_path / "not-a-zip.zip"
    archive.write_text("hello", encoding="utf-8")
    with pytest.raises(InvalidArchiveError):
        validate_archive(archive)


def test_validate_archive_missing_file(tmp_path):
    with pytest.raises(ArchiveNotFoundError):
        validate_archive(tmp_path / "missing.zip")


def test_is_cmapss_archive_false_for_unrelated_zip(tmp_path):
    archive = tmp_path / "other.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("file.txt", "x")
    assert is_cmapss_archive(archive) is False


def test_find_archive_none_when_missing(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    assert find_archive(raw_dir) is None


def test_find_archive_finds_expected_name(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    archive = _make_archive(tmp_path, layout="nested")
    archive.rename(raw_dir / "CMAPSSData.zip")
    assert find_archive(raw_dir) == raw_dir / "CMAPSSData.zip"


# --- extraction --------------------------------------------------------------


@pytest.mark.parametrize("layout", ["direct", "nested"])
def test_extract_dataset_supports_common_layouts(tmp_path, layout):
    archive = _make_archive(tmp_path, layout=layout)
    dest = extract_dataset(archive, tmp_path / "out" / "CMAPSSData")
    for name in ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]:
        assert (dest / name).is_file()


def test_extract_dataset_skips_when_already_present(tmp_path):
    archive = _make_archive(tmp_path)
    dest = extract_dataset(archive, tmp_path / "out" / "CMAPSSData")
    extracted = extract_dataset(archive, dest)
    assert extracted == dest


# --- dataset directory validation -------------------------------------------


def test_validate_dataset_directory_detects_subsets(tmp_path):
    _write_subset(tmp_path, "FD001")
    _write_subset(tmp_path, "FD004")
    subsets = validate_dataset_directory(tmp_path)
    assert subsets == {"FD001", "FD004"}


def test_validate_dataset_directory_missing_files(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "train_FD001.txt").write_text("x", encoding="utf-8")
    with pytest.raises(MissingFilesError) as excinfo:
        validate_dataset_directory(tmp_path)
    assert "test_FD001.txt" in str(excinfo.value)
    assert "RUL_FD001.txt" in str(excinfo.value)


# --- loading ----------------------------------------------------------------


def test_load_train_subset_schema(tmp_path):
    _write_subset(tmp_path)
    train = loader.load_train_subset("FD001", tmp_path)
    assert train.shape[1] == 26
    assert list(train.columns) == loader.COLUMNS


def test_load_test_subset_ignores_trailing_whitespace(tmp_path):
    _write_subset(tmp_path)
    test = loader.load_test_subset("FD001", tmp_path)
    assert test.shape[1] == 26
    assert list(test.columns) == loader.COLUMNS


def test_load_rul_subset(tmp_path):
    _write_subset(tmp_path)
    rul = loader.load_rul_subset("FD001", tmp_path)
    assert isinstance(rul, pd.Series)
    assert rul.name == "rul"
    assert rul.tolist() == [2, 1]


def test_load_unknown_subset(tmp_path):
    _write_subset(tmp_path)
    with pytest.raises(DataValidationError):
        loader.load_train_subset("FD999", tmp_path)


# --- RUL generation ---------------------------------------------------------


def test_add_rul_labels(tmp_path):
    _write_subset(tmp_path)
    train = loader.load_train_subset("FD001", tmp_path)
    labeled = add_rul_labels(train)
    assert labeled.loc[(labeled["unit_id"] == 1) & (labeled["cycle"] == 1), "rul"].iloc[0] == 4
    assert labeled.loc[(labeled["unit_id"] == 1) & (labeled["cycle"] == 5), "rul"].iloc[0] == 0
    assert labeled.loc[(labeled["unit_id"] == 2) & (labeled["cycle"] == 4), "rul"].iloc[0] == 0
    assert (labeled["rul"] >= 0).all()


def test_prepare_test_rul_alignment(tmp_path):
    _write_subset(tmp_path)
    test = loader.load_test_subset("FD001", tmp_path)
    rul = loader.load_rul_subset("FD001", tmp_path)
    aligned = prepare_test_rul(test, rul)
    assert aligned.loc[(aligned["unit_id"] == 1) & (aligned["cycle"] == 3), "rul"].iloc[0] == 2
    assert aligned.loc[(aligned["unit_id"] == 1) & (aligned["cycle"] == 1), "rul"].iloc[0] == 4
    assert aligned.loc[(aligned["unit_id"] == 2) & (aligned["cycle"] == 2), "rul"].iloc[0] == 1
    assert aligned.loc[(aligned["unit_id"] == 2) & (aligned["cycle"] == 1), "rul"].iloc[0] == 2


def test_prepare_test_rul_length_mismatch(tmp_path):
    _write_subset(tmp_path)
    test = loader.load_test_subset("FD001", tmp_path)
    with pytest.raises(DataValidationError):
        prepare_test_rul(test, pd.Series([1, 2, 3]))


# --- validation -------------------------------------------------------------


def test_validate_schema_rejects_wrong_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(DataValidationError):
        validate_schema(df)


def test_validate_schema_rejects_nan_required(tmp_path):
    _write_subset(tmp_path)
    df = loader.load_train_subset("FD001", tmp_path)
    df.loc[0, "unit_id"] = pd.NA
    with pytest.raises(DataValidationError):
        validate_schema(df)


def test_validate_engine_trajectories_rejects_duplicate_cycles(tmp_path):
    _write_subset(tmp_path)
    df = loader.load_train_subset("FD001", tmp_path)
    bad = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError):
        validate_engine_trajectories(bad)


def test_validate_engine_trajectories_rejects_negative_unit(tmp_path):
    _write_subset(tmp_path)
    df = loader.load_train_subset("FD001", tmp_path)
    df.loc[0, "unit_id"] = -1
    with pytest.raises(DataValidationError):
        validate_engine_trajectories(df)


def test_validate_rul_rejects_negative(tmp_path):
    _write_subset(tmp_path)
    df = add_rul_labels(loader.load_train_subset("FD001", tmp_path))
    df.loc[0, "rul"] = -1
    with pytest.raises(DataValidationError):
        validate_rul(df)


def test_validate_rul_rejects_nonzero_final(tmp_path):
    _write_subset(tmp_path)
    df = add_rul_labels(loader.load_train_subset("FD001", tmp_path))
    df.loc[(df["unit_id"] == 1) & (df["cycle"] == 5), "rul"] = 3
    with pytest.raises(DataValidationError):
        validate_rul(df)


def test_validate_rul_accepts_valid_train(tmp_path):
    _write_subset(tmp_path)
    df = add_rul_labels(loader.load_train_subset("FD001", tmp_path))
    validate_rul(df)


# --- end-to-end -------------------------------------------------------------


def test_preprocess_dataset_saves_parquet(tmp_path):
    raw = tmp_path / "raw" / "CMAPSSData"
    _write_subset(raw)
    processed = tmp_path / "processed"
    artifacts = preprocess_dataset("FD001", raw_dir=raw, processed_dir=processed)
    train_path, test_path = artifacts["train"], artifacts["test"]
    assert train_path.is_file()
    assert test_path.is_file()
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    assert list(train.columns) == loader.COLUMNS + ["rul"]
    assert list(test.columns) == loader.COLUMNS + ["rul"]
    assert train["rul"].min() == 0


@pytest.mark.integration
def test_integration_full_pipeline_on_real_dataset(tmp_path):
    archive = find_archive()
    if archive is None:
        pytest.skip("C-MAPSS archive not present in data/raw")
    raw = tmp_path / "raw"
    data_dir = extract_dataset(archive, raw / "CMAPSSData")
    subsets = validate_dataset_directory(data_dir)
    assert "FD001" in subsets
    processed = tmp_path / "processed"
    artifacts = preprocess_dataset("FD001", raw_dir=data_dir, processed_dir=processed)
    assert artifacts["train"].is_file()
    assert artifacts["test"].is_file()
