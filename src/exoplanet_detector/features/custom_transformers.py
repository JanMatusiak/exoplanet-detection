"""Custom sklearn transformers that wrap reusable preprocessing helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from exoplanet_detector.features.feature_selection import PHYSICAL_INTERVALS
from exoplanet_detector.features.outliers import (
    apply_iqr_clipping,
    apply_physical_outlier_screening,
    fit_iqr_fences,
)
from exoplanet_detector.features.preprocessing import (
    apply_left_skew_reflect_log1p,
    apply_right_skew_log1p,
    drop_feature_columns,
    fit_left_skew_reflection_max,
)

PhysicalInterval = tuple[float | None, float | None]
PhysicalIntervalMap = Mapping[str, PhysicalInterval]
IqrFence = tuple[float, float]
IqrFenceMap = Mapping[str, IqrFence]


def make_column_dropper(columns: Iterable[str]) -> FunctionTransformer:
    """Create a stateless column drop transformer."""
    return FunctionTransformer(
        func=drop_feature_columns,
        kw_args={"columns": list(columns)},
        validate=False,
    )


def make_right_skew_log_transformer(columns: Iterable[str]) -> FunctionTransformer:
    """Create a stateless right-skew log1p transformer."""
    return FunctionTransformer(
        func=apply_right_skew_log1p,
        kw_args={"columns": list(columns)},
        validate=False,
    )


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Backward-compatible wrapper around a FunctionTransformer column dropper."""

    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        self.transformer_ = make_column_dropper(self.columns)
        self.transformer_.fit(x, y)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["transformer_"])
        return self.transformer_.transform(x)


class RightSkewLogTransformer(BaseEstimator, TransformerMixin):
    """Backward-compatible wrapper around a FunctionTransformer log1p step."""

    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        self.transformer_ = make_right_skew_log_transformer(self.columns)
        self.transformer_.fit(x, y)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["transformer_"])
        return self.transformer_.transform(x)


class LeftSkewReflectLogTransformer(BaseEstimator, TransformerMixin):
    """Stateful reflected-log1p transformer for left-skewed columns."""

    def __init__(
        self,
        columns: Iterable[str],
        *,
        reflection_max: Mapping[str, float] | None = None,
        eps: float = 1e-6,
    ):
        self.columns = list(columns)
        self.reflection_max = dict(reflection_max) if reflection_max is not None else None
        self.eps = eps

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        if self.reflection_max is None:
            self.reflection_max_ = fit_left_skew_reflection_max(
                x,
                self.columns,
                eps=self.eps,
            )
        else:
            self.reflection_max_ = {key: float(value) for key, value in self.reflection_max.items()}
            missing_columns = [column for column in self.columns if column not in self.reflection_max_]
            if missing_columns:
                fitted_missing = fit_left_skew_reflection_max(
                    x,
                    missing_columns,
                    eps=self.eps,
                )
                self.reflection_max_.update(fitted_missing)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["reflection_max_"])
        transformed, _ = apply_left_skew_reflect_log1p(
            x,
            self.columns,
            reflection_max=self.reflection_max_,
            eps=self.eps,
        )
        return transformed


class PhysicalOutlierScreener(BaseEstimator, TransformerMixin):
    """Replace out-of-range values using physical feature intervals."""

    def __init__(
        self,
        *,
        intervals: PhysicalIntervalMap = PHYSICAL_INTERVALS,
        replace_with: float = float("nan"),
    ):
        self.intervals = dict(intervals)
        self.replace_with = replace_with

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        screened, summary = apply_physical_outlier_screening(
            x,
            intervals=self.intervals,
            replace_with=self.replace_with,
        )
        self.last_summary_ = summary
        return screened


class IqrClipper(BaseEstimator, TransformerMixin):
    """Fit train-time IQR fences and clip values to those fences."""

    def __init__(
        self,
        *,
        columns: Iterable[str] | None = None,
        whisker_width: float = 1.5,
        fences: IqrFenceMap | None = None,
    ):
        self.columns = list(columns) if columns is not None else None
        self.whisker_width = whisker_width
        self.fences = dict(fences) if fences is not None else None

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        if self.fences is not None:
            self.fences_ = {
                feature: (float(lower), float(upper))
                for feature, (lower, upper) in self.fences.items()
            }
            self.fit_summary_ = pd.DataFrame(
                [
                    {
                        "feature": feature,
                        "lower_fence": lower,
                        "upper_fence": upper,
                        "outlier_n": 0,
                        "outlier_pct": 0.0,
                    }
                    for feature, (lower, upper) in self.fences_.items()
                ]
            )
        else:
            self.fences_, self.fit_summary_ = fit_iqr_fences(
                x,
                columns=self.columns,
                whisker_width=self.whisker_width,
            )
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["fences_"])
        return apply_iqr_clipping(x, self.fences_)
