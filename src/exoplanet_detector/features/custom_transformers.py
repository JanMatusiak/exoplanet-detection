"""Custom sklearn transformers for notebook-derived feature preprocessing."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.drop(columns=self.columns, errors="ignore")


class RightSkewLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        transformed = x.copy()
        valid_columns = [column for column in self.columns if column in transformed.columns]
        transformed[valid_columns] = transformed[valid_columns].apply(
            lambda series: np.log1p(series.clip(lower=0))
        )
        return transformed


class LeftSkewReflectLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Iterable[str], eps: float = 1e-6):
        self.columns = list(columns)
        self.eps = eps
        self.reflection_max_: dict[str, float] = {}

    def fit(self, x: pd.DataFrame, y=None):  # noqa: D401, ANN001
        self.reflection_max_ = {}
        for column in self.columns:
            if column in x.columns:
                self.reflection_max_[column] = float(x[column].max(skipna=True) + self.eps)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        transformed = x.copy()
        for column in self.columns:
            if column not in transformed.columns:
                continue
            c = self.reflection_max_.get(column, float(transformed[column].max(skipna=True) + self.eps))
            reflected = c - transformed[column]
            transformed[column] = np.log1p(reflected.clip(lower=0))
        return transformed
