"""Reusable outlier screening and clipping helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from exoplanet_detector.features.feature_selection import PHYSICAL_INTERVALS

PhysicalInterval = tuple[float | None, float | None]
PhysicalIntervalMap = Mapping[str, PhysicalInterval]
IqrFence = tuple[float, float]
IqrFenceMap = Mapping[str, IqrFence]


def apply_physical_outlier_screening(
    df: pd.DataFrame,
    *,
    intervals: PhysicalIntervalMap = PHYSICAL_INTERVALS,
    replace_with: float = float("nan"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace values outside feature-specific physical bounds and report summary stats.

    Returns:
        (screened_dataframe, summary_dataframe)
    """
    screened = df.copy()
    summary_rows: list[dict[str, float | int | str | None]] = []

    for feature, (lower, upper) in intervals.items():
        if feature not in screened.columns:
            continue

        values = pd.to_numeric(screened[feature], errors="coerce")
        out_of_range = pd.Series(False, index=values.index)

        if lower is not None:
            out_of_range |= values < lower
        if upper is not None:
            out_of_range |= values > upper

        summary_rows.append(
            {
                "feature": feature,
                "lower": lower,
                "upper": upper,
                "out_of_range_n": int(out_of_range.sum()),
                "out_of_range_pct": float(out_of_range.mean()),
            }
        )
        screened.loc[out_of_range, feature] = replace_with

    summary = pd.DataFrame(summary_rows).sort_values("out_of_range_pct", ascending=False)
    return screened, summary


def fit_iqr_fences(
    df: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    whisker_width: float = 1.5,
) -> tuple[dict[str, IqrFence], pd.DataFrame]:
    """
    Fit per-feature IQR fences and return them with a diagnostics table.

    Returns:
        (fences_by_feature, summary_dataframe)
    """
    selected_columns = list(columns) if columns is not None else list(df.columns)
    fences: dict[str, IqrFence] = {}
    summary_rows: list[dict[str, float | int | str]] = []

    for feature in selected_columns:
        if feature not in df.columns:
            continue

        values = pd.to_numeric(df[feature], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue

        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - whisker_width * iqr
        upper = q3 + whisker_width * iqr
        outlier_mask = (values < lower) | (values > upper)

        fences[feature] = (float(lower), float(upper))
        summary_rows.append(
            {
                "feature": feature,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": float(lower),
                "upper_fence": float(upper),
                "outlier_n": int(outlier_mask.sum()),
                "outlier_pct": float(outlier_mask.mean()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("outlier_pct", ascending=False)
    return fences, summary


def apply_iqr_clipping(df: pd.DataFrame, fences: IqrFenceMap) -> pd.DataFrame:
    """Clip feature values to previously fitted IQR fences."""
    clipped = df.copy()
    for feature, (lower, upper) in fences.items():
        if feature not in clipped.columns:
            continue
        values = pd.to_numeric(clipped[feature], errors="coerce")
        clipped[feature] = values.clip(lower=lower, upper=upper)
    return clipped


def fit_iqr_fences_and_clip(
    df: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    whisker_width: float = 1.5,
) -> tuple[pd.DataFrame, dict[str, IqrFence], pd.DataFrame]:
    """
    Fit train-time IQR fences and clip values in one step.

    Returns:
        (clipped_dataframe, fences_by_feature, summary_dataframe)
    """
    fences, summary = fit_iqr_fences(df, columns=columns, whisker_width=whisker_width)
    clipped = apply_iqr_clipping(df, fences)
    return clipped, fences, summary
