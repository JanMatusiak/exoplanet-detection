"""Raw-data loading and harmonization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from exoplanet_detector.config import (
    CANDIDATE_LABEL,
    K2P_DEFAULT_FLAG_COLUMN,
    K2P_DEFAULT_FLAG_VALUE,
    K2P_RAW_FILE,
    KOI_RAW_FILE,
    LABEL_MAP,
    TARGET_COLUMN,
)
from exoplanet_detector.features.feature_selection import (
    BASE_DROP_COLUMNS,
    K2P_PHYSICAL_COLUMNS_SET,
    K2P_RENAME_MAP,
    KOI_PHYSICAL_COLUMNS_SET,
    KOI_RENAME_MAP,
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def load_koi_full(path: Path = KOI_RAW_FILE) -> pd.DataFrame:
    return _read_csv(path)


def load_k2p_full(path: Path = K2P_RAW_FILE, *, default_only: bool = True) -> pd.DataFrame:
    df = _read_csv(path)
    if default_only and K2P_DEFAULT_FLAG_COLUMN in df.columns:
        return df[df[K2P_DEFAULT_FLAG_COLUMN] == K2P_DEFAULT_FLAG_VALUE].copy()
    return df


def select_and_rename_columns(
    df: pd.DataFrame,
    columns_set: Iterable[str],
    rename_map: Mapping[str, str],
) -> pd.DataFrame:
    columns = list(columns_set)
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        missing_cols = ", ".join(missing)
        raise KeyError(f"Missing required columns: {missing_cols}")
    return df.loc[:, columns].rename(columns=rename_map).copy()


def process_dataset(
    df: pd.DataFrame,
    columns_set: Iterable[str],
    rename_map: Mapping[str, str],
) -> pd.DataFrame:
    """Backward-compatible wrapper for notebook prototype code."""
    return select_and_rename_columns(df, columns_set, rename_map)


def split_labeled_and_candidates(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    candidate_label: str = CANDIDATE_LABEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = df[df[target_column].isin(LABEL_MAP)].copy()
    candidates = df[df[target_column] == candidate_label].copy()
    return labeled, candidates


def map_labels(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    label_map: Mapping[str, int] = LABEL_MAP,
) -> pd.DataFrame:
    mapped = df.copy()
    mapped[target_column] = mapped[target_column].map(label_map)
    mapped = mapped[mapped[target_column].notna()].copy()
    mapped[target_column] = mapped[target_column].astype(int)
    return mapped


def convert_transit_depth_percent_to_ppm(
    df: pd.DataFrame,
    *,
    transit_depth_column: str = "transit_depth",
) -> pd.DataFrame:
    converted = df.copy()
    converted[transit_depth_column] = (
        pd.to_numeric(converted[transit_depth_column], errors="coerce") * 10000.0
    )
    return converted


def drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return df.drop(columns=list(columns), errors="ignore")


def prepare_harmonized_datasets(
    koi_df: pd.DataFrame | None = None,
    k2p_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Prepare KOI and K2P datasets following notebook 01 decisions."""
    koi_raw = load_koi_full() if koi_df is None else koi_df
    k2p_raw = load_k2p_full(default_only=True) if k2p_df is None else k2p_df

    koi_harmonized = select_and_rename_columns(koi_raw, KOI_PHYSICAL_COLUMNS_SET, KOI_RENAME_MAP)
    k2p_harmonized = select_and_rename_columns(k2p_raw, K2P_PHYSICAL_COLUMNS_SET, K2P_RENAME_MAP)

    koi_labeled, koi_candidates = split_labeled_and_candidates(koi_harmonized)
    k2p_labeled, k2p_candidates = split_labeled_and_candidates(k2p_harmonized)

    koi_labeled = map_labels(koi_labeled)
    k2p_labeled = map_labels(k2p_labeled)

    k2p_labeled = convert_transit_depth_percent_to_ppm(k2p_labeled)
    k2p_candidates = convert_transit_depth_percent_to_ppm(k2p_candidates)

    koi_labeled = drop_columns(koi_labeled, BASE_DROP_COLUMNS)
    k2p_labeled = drop_columns(k2p_labeled, BASE_DROP_COLUMNS)

    return {
        "koi_labeled": koi_labeled,
        "koi_candidates": koi_candidates,
        "k2p_labeled": k2p_labeled,
        "k2p_candidates": k2p_candidates,
    }
