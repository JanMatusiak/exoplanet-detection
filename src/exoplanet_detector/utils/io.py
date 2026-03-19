"""Generic project I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from exoplanet_detector.config import K2P_FILE, KOI_TEST_FILE, KOI_TRAIN_FILE, PROCESSED_DATA_DIR


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    ensure_directory(path.parent)
    df.to_csv(path, index=index)
    return path


def load_dataframe(path: Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **read_csv_kwargs)


def save_json(data: Any, path: Path, *, indent: int = 2) -> Path:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent)
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_artifact(obj: Any, path: Path) -> Path:
    ensure_directory(path.parent)
    joblib.dump(obj, path)
    return path


def load_artifact(path: Path) -> Any:
    return joblib.load(path)


def save_processed_splits(
    koi_train_set: pd.DataFrame,
    koi_test_set: pd.DataFrame,
    k2p_set: pd.DataFrame,
    *,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    koi_train_path = output_dir / KOI_TRAIN_FILE.name
    koi_test_path = output_dir / KOI_TEST_FILE.name
    k2p_path = output_dir / K2P_FILE.name

    save_dataframe(koi_train_set, koi_train_path)
    save_dataframe(koi_test_set, koi_test_path)
    save_dataframe(k2p_set, k2p_path)

    return {
        "koi_train_set": koi_train_path,
        "koi_test_set": koi_test_path,
        "k2p_set": k2p_path,
    }
