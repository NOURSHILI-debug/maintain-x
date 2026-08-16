from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

EXPECTED_ARCHIVE_NAME = "CMAPSSData.zip"
EXPECTED_ARCHIVE = RAW_DIR / EXPECTED_ARCHIVE_NAME
CMAPSS_DIR = RAW_DIR / "CMAPSSData"
