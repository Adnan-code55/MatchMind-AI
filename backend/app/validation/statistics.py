"""
Feature statistics utilities for validation reporting.

This module provides numeric statistics, correlation analysis, and target
distribution computations for engineered datasets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.logger import PipelineLogger


@dataclass
class StatisticsCache:
    """Lightweight in-memory cache for validation statistics."""

    _cache: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value by key."""
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Cache a computed value by key."""
        self._cache[key] = value

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()


class ValidationStatistics:
    """Compute validation statistics for engineered datasets."""

    def __init__(self, cache: Optional[StatisticsCache] = None) -> None:
        """Initialize validation statistics with optional cache injection."""
        self.cache = cache or StatisticsCache()
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)
        self.logger.info("ValidationStatistics initialized")

    def numeric_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute numeric summary statistics for dataset columns."""
        key = f"numeric_statistics:{id(df)}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        numeric_df = df.select_dtypes(include=[np.number]).copy()
        summary = numeric_df.describe().transpose()
        summary["missing"] = numeric_df.isna().sum()
        summary["dtype"] = numeric_df.dtypes.astype(str)
        result = summary.round(6).where(~summary.isna(), None).to_dict(orient="index")

        self.cache.set(key, result)
        return result

    def correlation_matrix(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute the correlation matrix for numeric features."""
        key = f"correlation_matrix:{id(df)}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if numeric_df.empty:
            return {}

        matrix = numeric_df.corr().round(6)
        result = matrix.where(~matrix.isna(), None).to_dict()
        self.cache.set(key, result)
        return result

    def highly_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.9,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return highly correlated feature pairs above a threshold."""
        key = f"highly_correlated:{id(df)}:{threshold}:{limit}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if numeric_df.shape[1] < 2:
            return []

        matrix = numeric_df.corr().abs()
        pairs: List[Tuple[str, str, float]] = []

        for i, feature in enumerate(matrix.columns):
            for other in matrix.columns[i + 1 :]:
                value = float(matrix.at[feature, other])
                if value >= threshold:
                    pairs.append((feature, other, value))

        pairs.sort(key=lambda item: item[2], reverse=True)
        result = [
            {"feature_a": a, "feature_b": b, "correlation": corr}
            for a, b, corr in pairs[:limit]
        ]

        self.cache.set(key, result)
        return result

    def target_distribution(
        self, df: pd.DataFrame, target_column: str
    ) -> Dict[str, Any]:
        """Compute target distribution counts and ratios."""
        key = f"target_distribution:{id(df)}:{target_column}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' does not exist in dataset.")

        values = df[target_column].value_counts(dropna=False)
        ratios = df[target_column].value_counts(dropna=False, normalize=True).round(6)
        result = {
            "counts": values.to_dict(),
            "ratios": ratios.to_dict(),
            "dtype": str(df[target_column].dtype),
        }

        self.cache.set(key, result)
        return result

    def feature_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute a concise summary for dataset features."""
        key = f"feature_summary:{id(df)}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        summary = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "count": df.count(),
                "unique": df.nunique(dropna=False),
                "missing": df.isna().sum(),
            }
        )

        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            numeric_stats = numeric_df.agg(["min", "max", "mean", "std"]).transpose()
            summary = summary.join(numeric_stats)

        result = summary.round(6).where(~summary.isna(), None).to_dict(orient="index")
        self.cache.set(key, result)
        return result
