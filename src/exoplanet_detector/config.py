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
K2P_FILE = PROCESSED_DATA_DIR / "K2P_labeled_set.csv"
KOI_UNLABELED_FILE = PROCESSED_DATA_DIR / "KOI_unlabeled_set.csv"
K2P_UNLABELED_FILE = PROCESSED_DATA_DIR / "K2P_unlabeled_set.csv"

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

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_SEARCH_ARTIFACTS_DIR = ARTIFACTS_DIR / "model_search"
EVALUATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "evaluation"
DEPLOYMENT_ARTIFACTS_DIR = ARTIFACTS_DIR / "deployment"

DEFAULT_RUN_TAG = "v1"
MIN_PRECISION_FLOOR = 0.5
MIN_RECALL_FLOOR = 0.5


def get_run_artifact_dirs(run_tag: str = DEFAULT_RUN_TAG, *, create: bool = False) -> dict[str, Path]:
    """
    Return model-search/evaluation/deployment artifact directories for a run tag.

    Args:
        run_tag: Shared experiment/version identifier (e.g., ``v1``, ``v2``).
        create: If True, create the directories if they do not exist.
    """
    directories = {
        "model_search": MODEL_SEARCH_ARTIFACTS_DIR / run_tag,
        "evaluation": EVALUATION_ARTIFACTS_DIR / run_tag,
        "deployment": DEPLOYMENT_ARTIFACTS_DIR / run_tag,
    }
    if create:
        for path in directories.values():
            path.mkdir(parents=True, exist_ok=True)
    return directories
