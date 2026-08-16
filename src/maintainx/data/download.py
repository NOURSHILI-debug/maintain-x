import io
import shutil
import tempfile
import zipfile
from pathlib import Path

from maintainx.data import paths
from maintainx.data.errors import (
    ArchiveNotFoundError,
    CMAPSSDataError,
    InvalidArchiveError,
)
from maintainx.data.validation import validate_dataset_directory

REQUIRED_BASENAMES: set[str] = {"train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"}

ARCHIVE_MISSING_MESSAGE = (
    "C-MAPSS archive not found.\n"
    "\n"
    "The NASA C-MAPSS turbofan engine degradation dataset must be downloaded manually:\n"
    "  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/\n"
    "\n"
    f"Place the archive at:\n"
    f"  {paths.EXPECTED_ARCHIVE}\n"
    "\n"
    "Then run this command again."
)


def _collect_member_names(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def is_cmapss_archive(path: Path) -> bool:
    """Return True if the ZIP contains the C-MAPSS files directly or in a nested archive."""
    if not zipfile.is_zipfile(path):
        return False
    try:
        names = _collect_member_names(path)
        if REQUIRED_BASENAMES <= {Path(name).name for name in names}:
            return True
        for name in names:
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(path).open(name) as inner:
                    inner_names = zipfile.ZipFile(io.BytesIO(inner.read())).namelist()
                if REQUIRED_BASENAMES <= {Path(inner_name).name for inner_name in inner_names}:
                    return True
    except (zipfile.BadZipFile, KeyError, OSError):
        return False
    return False


def find_archive(raw_dir: Path | None = None) -> Path | None:
    """Locate the C-MAPSS archive in the raw data directory, or return None."""
    directory = Path(raw_dir) if raw_dir is not None else paths.RAW_DIR
    if not directory.is_dir():
        return None
    expected = directory / paths.EXPECTED_ARCHIVE_NAME
    if expected.is_file():
        return expected
    candidates = [p for p in sorted(directory.glob("*.zip")) if is_cmapss_archive(p)]
    if len(candidates) > 1:
        raise CMAPSSDataError(
            f"Multiple C-MAPSS archives found in {directory}. "
            f"Keep exactly one and rename it to {paths.EXPECTED_ARCHIVE_NAME}."
        )
    return candidates[0] if candidates else None


def validate_archive(archive: Path) -> None:
    """Verify that an archive exists and is a valid C-MAPSS ZIP file."""
    if not Path(archive).is_file():
        raise ArchiveNotFoundError(f"Archive not found: {archive}")
    if not zipfile.is_zipfile(archive):
        raise InvalidArchiveError(f"'{archive}' is not a valid ZIP file")
    if not is_cmapss_archive(archive):
        raise InvalidArchiveError(
            f"'{archive}' does not contain the expected C-MAPSS files "
            f"({', '.join(sorted(REQUIRED_BASENAMES))})."
        )


def _extract_archive(archive: Path, dest: Path) -> None:
    """Recursively extract a ZIP, safely handling nested ZIP members and zip-slip."""
    with zipfile.ZipFile(archive) as zf:
        bad_entry = zf.testzip()
        if bad_entry is not None:
            raise InvalidArchiveError(f"Archive is corrupt (bad entry: {bad_entry})")
        dest_resolved = dest.resolve()
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest_resolved):
                raise InvalidArchiveError(f"Unsafe path in archive: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            if member.filename.lower().endswith(".zip"):
                _extract_archive(target, dest)
                target.unlink()


def extract_dataset(archive: Path, dest_dir: Path | None = None) -> Path:
    """Extract the C-MAPSS archive into a flat directory, skipping files that exist."""
    destination = Path(dest_dir) if dest_dir is not None else paths.CMAPSS_DIR
    validate_archive(archive)
    if destination.is_dir():
        present = {p.name for p in destination.glob("*.txt")}
        if REQUIRED_BASENAMES <= present:
            print(f"Already extracted; skipping extraction to {destination}")
            return destination
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cmapss-extract-") as tmp:
        tmp_root = Path(tmp)
        _extract_archive(archive, tmp_root)
        data_files = sorted(tmp_root.rglob("*.txt"))
        copied = 0
        skipped = 0
        for src in data_files:
            target = destination / src.name
            if target.exists():
                skipped += 1
            else:
                shutil.copy2(src, target)
                copied += 1
        print(f"Extracted {len(data_files)} text file(s): {copied} new, {skipped} already present")
    return destination


def main() -> int:
    try:
        archive = find_archive()
    except CMAPSSDataError as exc:
        print(f"ERROR: {exc}")
        return 1

    if archive is None:
        print(ARCHIVE_MISSING_MESSAGE)
        return 1

    print(f"Archive found: {archive}")
    if archive.name != paths.EXPECTED_ARCHIVE_NAME:
        print(f"Note: expected '{paths.EXPECTED_ARCHIVE_NAME}'; using '{archive.name}' instead.")

    try:
        validate_archive(archive)
        destination = extract_dataset(archive)
        subsets = validate_dataset_directory(destination)
    except CMAPSSDataError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Archive is a valid C-MAPSS ZIP.")
    print(f"Extraction succeeded. Files are in: {destination}")
    print(f"Expected files found. Complete subsets detected: {', '.join(sorted(subsets))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
