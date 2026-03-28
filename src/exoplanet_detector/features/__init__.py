"""Feature definitions and preprocessing utilities."""

from exoplanet_detector.features.feature_selection import (
    ANALYSIS_DROP_COLUMNS,
    BASE_DROP_COLUMNS,
    CONSTANT_VALUE_DROP_COLUMNS,
    CORRELATION_DROP_COLUMNS,
    EXPLORATORY_RIGHT_SKEWED_COLUMNS,
    FINAL_FEATURE_COLUMNS,
    K2P_PHYSICAL_COLUMNS_SET,
    K2P_RENAME_MAP,
    KOI_PHYSICAL_COLUMNS_SET,
    KOI_RENAME_MAP,
    LEFT_SKEWED_COLUMNS,
    PHYSICAL_INTERVALS,
    RIGHT_SKEWED_COLUMNS,
)
from exoplanet_detector.features.preprocessing import (
    apply_left_skew_reflect_log1p,
    apply_right_skew_log1p,
    drop_feature_columns,
    feature_summary,
    fit_left_skew_reflection_max,
    preprocess_feature_frame,
)
from exoplanet_detector.features.outliers import (
    apply_iqr_clipping,
    apply_physical_outlier_screening,
    fit_iqr_fences,
    fit_iqr_fences_and_clip,
)
from exoplanet_detector.features.pipelines import build_preprocessing_pipeline
from exoplanet_detector.features.custom_transformers import (
    ColumnDropper,
    FinalFeatureSelector,
    IqrClipper,
    LeftSkewReflectLogTransformer,
    PhysicalOutlierScreener,
    RightSkewLogTransformer,
)

__all__ = [
    "ANALYSIS_DROP_COLUMNS",
    "BASE_DROP_COLUMNS",
    "CONSTANT_VALUE_DROP_COLUMNS",
    "CORRELATION_DROP_COLUMNS",
    "EXPLORATORY_RIGHT_SKEWED_COLUMNS",
    "FINAL_FEATURE_COLUMNS",
    "K2P_PHYSICAL_COLUMNS_SET",
    "K2P_RENAME_MAP",
    "KOI_PHYSICAL_COLUMNS_SET",
    "KOI_RENAME_MAP",
    "LEFT_SKEWED_COLUMNS",
    "PHYSICAL_INTERVALS",
    "RIGHT_SKEWED_COLUMNS",
    "apply_left_skew_reflect_log1p",
    "apply_right_skew_log1p",
    "drop_feature_columns",
    "feature_summary",
    "fit_left_skew_reflection_max",
    "preprocess_feature_frame",
    "apply_iqr_clipping",
    "apply_physical_outlier_screening",
    "fit_iqr_fences",
    "fit_iqr_fences_and_clip",
    "build_preprocessing_pipeline",
    "ColumnDropper",
    "FinalFeatureSelector",
    "RightSkewLogTransformer",
    "LeftSkewReflectLogTransformer",
    "PhysicalOutlierScreener",
    "IqrClipper",
]
