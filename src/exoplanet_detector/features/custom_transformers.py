"""Custom sklearn transformers that wrap reusable preprocessing helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop selected columns from a pandas DataFrame."""

    def __init__(self, columns: Iterable[str], *, strict: bool = False):
        self.columns = columns
        self.strict = strict

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        if not isinstance(x, pd.DataFrame):
            raise TypeError("`ColumnDropper` expects a pandas DataFrame as input.")

        columns = list(self.columns)
        missing_columns = [column for column in columns if column not in x.columns]
        if self.strict and missing_columns:
            missing_csv = ", ".join(missing_columns)
            raise KeyError(f"Missing columns to drop: {missing_csv}")

        self.columns_to_drop_ = [column for column in columns if column in x.columns]
        self.missing_columns_ = missing_columns
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["columns_to_drop_"])
        if not isinstance(x, pd.DataFrame):
            raise TypeError("`ColumnDropper` expects a pandas DataFrame as input.")
        return drop_feature_columns(x, self.columns_to_drop_)


class FinalFeatureSelector(BaseEstimator, TransformerMixin):
    """Select only the final feature columns expected by the modeling pipeline."""

    def __init__(self, columns: Iterable[str], *, strict: bool = False):
        self.columns = columns
        self.strict = strict

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        if not isinstance(x, pd.DataFrame):
            raise TypeError("`FinalFeatureSelector` expects a pandas DataFrame as input.")

        columns = list(self.columns)
        missing_columns = [column for column in columns if column not in x.columns]
        if self.strict and missing_columns:
            missing_csv = ", ".join(missing_columns)
            raise KeyError(f"Missing required final feature columns: {missing_csv}")

        self.selected_columns_ = [column for column in columns if column in x.columns]
        self.missing_columns_ = missing_columns
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["selected_columns_"])
        if not isinstance(x, pd.DataFrame):
            raise TypeError("`FinalFeatureSelector` expects a pandas DataFrame as input.")
        return x.loc[:, self.selected_columns_].copy()


class RightSkewLogTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to selected right-skewed columns."""

    def __init__(self, columns: Iterable[str], *, strict: bool = False):
        self.columns = columns
        self.strict = strict

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        if not isinstance(x, pd.DataFrame):
            raise TypeError("`RightSkewLogTransformer` expects a pandas DataFrame as input.")

        columns = list(self.columns)
        missing_columns = [column for column in columns if column not in x.columns]
        if self.strict and missing_columns:
            missing_csv = ", ".join(missing_columns)
            raise KeyError(f"Missing right-skew columns: {missing_csv}")

        self.columns_to_transform_ = [column for column in columns if column in x.columns]
        self.missing_columns_ = missing_columns
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["columns_to_transform_"])
        if not isinstance(x, pd.DataFrame):
            raise TypeError("`RightSkewLogTransformer` expects a pandas DataFrame as input.")
        return apply_right_skew_log1p(x, self.columns_to_transform_)


class LeftSkewReflectLogTransformer(BaseEstimator, TransformerMixin):
    """Stateful reflected-log1p transformer for left-skewed columns."""

    def __init__(
        self,
        columns: Iterable[str],
        *,
        reflection_max: Mapping[str, float] | None = None,
        eps: float = 1e-6,
    ):
        self.columns = columns
        self.reflection_max = reflection_max
        self.eps = eps

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        self.columns_ = list(self.columns)
        if self.reflection_max is None:
            self.reflection_max_ = fit_left_skew_reflection_max(
                x,
                self.columns_,
                eps=self.eps,
            )
        else:
            self.reflection_max_ = {key: float(value) for key, value in self.reflection_max.items()}
            missing_columns = [column for column in self.columns_ if column not in self.reflection_max_]
            if missing_columns:
                fitted_missing = fit_left_skew_reflection_max(
                    x,
                    missing_columns,
                    eps=self.eps,
                )
                self.reflection_max_.update(fitted_missing)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["columns_", "reflection_max_"])
        transformed, _ = apply_left_skew_reflect_log1p(
            x,
            self.columns_,
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
        self.intervals = intervals
        self.replace_with = replace_with

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        self.intervals_ = dict(self.intervals)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["intervals_"])
        screened, summary = apply_physical_outlier_screening(
            x,
            intervals=self.intervals_,
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
        self.columns = columns
        self.whisker_width = whisker_width
        self.fences = fences

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        selected_columns = list(self.columns) if self.columns is not None else None
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
                columns=selected_columns,
                whisker_width=self.whisker_width,
            )
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["fences_"])
        return apply_iqr_clipping(x, self.fences_)
