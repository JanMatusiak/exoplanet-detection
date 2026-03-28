"""Reusable preprocessing helpers extracted from notebook 02."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the notebook summary table for numeric feature inspection."""
    summary_rows: list[list[float | int | str]] = []
    for column in df.columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        non_null = numeric.dropna()

        if non_null.empty:
            summary_rows.append(
                [column, len(numeric), 1.0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            )
            continue

        value_counts = non_null.value_counts()
        top_fraction = value_counts.iloc[0] / len(non_null) if len(value_counts) else np.nan
        q01, q25, q50, q75, q99 = np.percentile(non_null, [1, 25, 50, 75, 99])

        summary_rows.append(
            [
                column,
                len(numeric),
                numeric.isna().mean(),
                non_null.nunique(),
                top_fraction,
                q01,
                q25,
                q50,
                q75,
                q99,
                q75 - q25,
            ]
        )

    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "feature",
            "n",
            "missing_rate",
            "n_unique",
            "top_value_frac",
            "p01",
            "p25",
            "median",
            "p75",
            "p99",
            "IQR",
        ],
    )
    return summary.sort_values(["missing_rate", "top_value_frac"], ascending=[True, False])


def drop_feature_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return df.drop(columns=list(columns), errors="ignore")


def apply_right_skew_log1p(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    transformed = df.copy()
    valid_columns = [column for column in columns if column in transformed.columns]
    transformed[valid_columns] = transformed[valid_columns].apply(
        lambda series: np.log1p(series.clip(lower=0))
    )
    return transformed


def fit_left_skew_reflection_max(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    eps: float = 1e-6,
) -> dict[str, float]:
    reflection_max: dict[str, float] = {}
    for column in columns:
        if column not in df.columns:
            continue
        reflection_max[column] = float(df[column].max(skipna=True) + eps)
    return reflection_max


def apply_left_skew_reflect_log1p(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    reflection_max: Mapping[str, float] | None = None,
    eps: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, float]]:
    transformed = df.copy()
    used_reflection_max = dict(reflection_max) if reflection_max is not None else {}

    for column in columns:
        if column not in transformed.columns:
            continue
        if column not in used_reflection_max:
            used_reflection_max[column] = float(transformed[column].max(skipna=True) + eps)
        reflected = used_reflection_max[column] - transformed[column]
        transformed[column] = np.log1p(reflected.clip(lower=0))

    return transformed, used_reflection_max
