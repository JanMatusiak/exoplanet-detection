"""Prebuilt sklearn preprocessing pipelines for model training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from exoplanet_detector.features.custom_transformers import (
    FinalFeatureSelector,
    IqrClipper,
    LeftSkewReflectLogTransformer,
    PhysicalOutlierScreener,
    RightSkewLogTransformer,
)
from exoplanet_detector.features.feature_selection import (
    FINAL_FEATURE_COLUMNS,
    LEFT_SKEWED_COLUMNS,
    PHYSICAL_INTERVALS,
    RIGHT_SKEWED_COLUMNS,
)

PhysicalInterval = tuple[float | None, float | None]
PhysicalIntervalMap = Mapping[str, PhysicalInterval]
IqrFence = tuple[float, float]
IqrFenceMap = Mapping[str, IqrFence]


def build_preprocessing_pipeline(
    *,
    final_feature_columns: Iterable[str] = FINAL_FEATURE_COLUMNS,
    right_skewed_columns: Iterable[str] = RIGHT_SKEWED_COLUMNS,
    left_skewed_columns: Iterable[str] = LEFT_SKEWED_COLUMNS,
    physical_intervals: PhysicalIntervalMap = PHYSICAL_INTERVALS,
    reflection_max: Mapping[str, float] | None = None,
    eps: float = 1e-6,
    whisker_width: float = 1.5,
    iqr_fences: IqrFenceMap | None = None,
    imputation_strategy: str = "median",
    with_scaling: bool = True,
) -> Pipeline:
    """
    Build the baseline preprocessing pipeline used before model fitting.

    Includes:
    - final feature selection
    - physical-range screening
    - right-skew log1p transform
    - left-skew reflected log1p transform
    - IQR clipping
    - missing-value imputation
    - feature scaling
    """
    selected_features = list(final_feature_columns)

    return Pipeline(
        steps=[
            (
                "select_final_features",
                FinalFeatureSelector(columns=selected_features, strict=True),
            ),
            (
                "physical_screen",
                PhysicalOutlierScreener(intervals=physical_intervals),
            ),
            ("right_log", RightSkewLogTransformer(columns=right_skewed_columns, strict=True)),
            (
                "left_log",
                LeftSkewReflectLogTransformer(
                    columns=left_skewed_columns,
                    reflection_max=reflection_max,
                    eps=eps,
                    strict=True,
                ),
            ),
            (
                "iqr_clip",
                IqrClipper(
                    columns=selected_features,
                    whisker_width=whisker_width,
                    fences=iqr_fences,
                ),
            ),
            ("impute", SimpleImputer(strategy=imputation_strategy)),
            ("scale", StandardScaler() if with_scaling else "passthrough"),
        ]
    )
