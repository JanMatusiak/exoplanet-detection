"""Data split helpers based on notebook 01."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from exoplanet_detector.config import (
    DEFAULT_FOLD_INDEX,
    GROUP_COLUMN,
    N_SPLITS,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from exoplanet_detector.data.load_data import prepare_harmonized_datasets


def stratified_group_train_test_split(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    group_column: str = GROUP_COLUMN,
    n_splits: int = N_SPLITS,
    fold_index: int = DEFAULT_FOLD_INDEX,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split with StratifiedGroupKFold to avoid host-level leakage."""
    if target_column not in df.columns or group_column not in df.columns:
        raise KeyError(f"`{target_column}` and `{group_column}` must exist in the dataframe.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    x = df.drop(columns=[target_column, group_column])
    y = df[target_column]
    groups = df[group_column]
    splits = list(splitter.split(x, y, groups=groups))

    if not 0 <= fold_index < len(splits):
        raise ValueError(
            f"`fold_index` must be in [0, {len(splits) - 1}], got {fold_index}."
        )

    train_idx, test_idx = splits[fold_index]
    train_df = df.iloc[train_idx, :].copy()
    test_df = df.iloc[test_idx, :].copy()
    return train_df, test_df


def create_processed_splits(
    koi_labeled: pd.DataFrame,
    k2p_labeled: pd.DataFrame,
    *,
    fold_index: int = DEFAULT_FOLD_INDEX,
) -> dict[str, pd.DataFrame]:
    koi_train_set, koi_test_set = stratified_group_train_test_split(
        koi_labeled, fold_index=fold_index
    )
    return {
        "koi_train_set": koi_train_set,
        "koi_test_set": koi_test_set,
        "k2p_set": k2p_labeled.copy(),
    }


def create_processed_splits_from_raw(*, fold_index: int = DEFAULT_FOLD_INDEX) -> dict[str, pd.DataFrame]:
    prepared = prepare_harmonized_datasets()
    return create_processed_splits(
        prepared["koi_labeled"],
        prepared["k2p_labeled"],
        fold_index=fold_index,
    )
