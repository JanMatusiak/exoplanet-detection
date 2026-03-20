"""Project-wide configuration constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

KOI_RAW_FILE = RAW_DATA_DIR / "KOI_full.csv"
K2P_RAW_FILE = RAW_DATA_DIR / "K2P_full.csv"

KOI_TRAIN_FILE = PROCESSED_DATA_DIR / "KOI_train_set.csv"
KOI_TEST_FILE = PROCESSED_DATA_DIR / "KOI_test_set.csv"
K2P_FILE = PROCESSED_DATA_DIR / "K2P_set.csv"

RANDOM_STATE = 42
N_SPLITS = 5
DEFAULT_FOLD_INDEX = 0

TARGET_COLUMN = "label"
GROUP_COLUMN = "group_id"

LABEL_MAP = {
    "CONFIRMED": 1,
    "FALSE POSITIVE": 0,
    "REFUTED": 0,
}

CANDIDATE_LABEL = "CANDIDATE"

K2P_DEFAULT_FLAG_COLUMN = "default_flag"
K2P_DEFAULT_FLAG_VALUE = 1
