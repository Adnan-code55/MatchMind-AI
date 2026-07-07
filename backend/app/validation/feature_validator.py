"""
Feature validation engine for MatchMind AI.

This module validates an engineered dataset prior to model training by checking
for missing or infinite values, duplicate rows and columns, constant and
near-constant features, invalid feature types, and target distribution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.logger import PipelineLogger
from .dataset_report import DatasetValidationReport
from .statistics import ValidationStatistics


class FeatureValidationError(Exception):
    """Base exception for feature validation failures."""


class FeatureValidator:
    """Validate engineered datasets before model training."""

    def __init__(
        self,
        statistics: Optional[ValidationStatistics] = None,
        near_constant_threshold: float = 0.01,
        correlation_threshold: float = 0.9,
    ) -> None:
        """Initialize the validator with optional dependencies."""
        self.statistics = statistics or ValidationStatistics()
        self.near_constant_threshold = near_constant_threshold
        self.correlation_threshold = correlation_threshold
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)
        self.logger.info("FeatureValidator initialized")

    def validate(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> DatasetValidationReport:
        """Validate the dataset and generate a structured validation report."""
        if not isinstance(df, pd.DataFrame):
            raise FeatureValidationError("Input must be a pandas DataFrame.")

        self.logger.info("Starting feature validation")
        rows, columns = df.shape
        feature_columns = [col for col in df.columns if col != target_column]
        feature_df = df[feature_columns].copy()

        invalid_feature_types = self._find_invalid_feature_types(feature_df)
        missing_values = self._detect_missing_values(feature_df)
        infinite_values = self._detect_infinite_values(feature_df)
        duplicate_rows = self._detect_duplicate_rows(df)
        duplicate_columns = self._detect_duplicate_columns(feature_df)
        constant_features = self._detect_constant_features(feature_df)
        near_constant_features = self._detect_near_constant_features(feature_df)
        target_distribution = self._detect_target_distribution(df, target_column)
        feature_summary = self.statistics.feature_summary(feature_df)
        numeric_statistics = self.statistics.numeric_statistics(feature_df)
        correlation_matrix = self.statistics.correlation_matrix(feature_df)
        highly_correlated_features = self.statistics.highly_correlated_features(
            feature_df,
            threshold=self.correlation_threshold,
        )

        warnings: List[str] = []
        recommendations: List[str] = []

        if missing_values:
            warnings.append(
                f"Found missing values in columns: {sorted(missing_values.keys())}"
            )
            recommendations.append("Impute or remove missing values before training.")

        if infinite_values:
            warnings.append(
                f"Found infinite values in columns: {sorted(infinite_values.keys())}"
            )
            recommendations.append("Replace infinite values with finite representations.")

        if duplicate_rows > 0:
            warnings.append(f"Found {duplicate_rows} duplicate row(s).")
            recommendations.append("Remove duplicate rows from the dataset.")

        if duplicate_columns:
            warnings.append(f"Found duplicate columns: {duplicate_columns}")
            recommendations.append("Remove or merge duplicate features.")

        if constant_features:
            warnings.append(f"Found constant features: {constant_features}")
            recommendations.append("Drop constant features to reduce model complexity.")

        if near_constant_features:
            warnings.append(
                f"Found near-constant features: {list(near_constant_features.keys())}"
            )
            recommendations.append(
                "Review near-constant features and consider removing them."
            )

        if invalid_feature_types:
            warnings.append(
                f"Found invalid feature types: {invalid_feature_types}"
            )
            recommendations.append(
                "Convert non-numeric features to numeric values or remove them."
            )

        if target_column and target_distribution:
            self._analyze_target_distribution(target_distribution, warnings)

        is_valid = not any(
            [
                bool(missing_values),
                bool(infinite_values),
                duplicate_rows > 0,
                bool(duplicate_columns),
                bool(constant_features),
                bool(near_constant_features),
                bool(invalid_feature_types),
            ]
        )

        report = DatasetValidationReport(
            rows=rows,
            columns=columns,
            feature_count=len(feature_columns),
            target_column=target_column,
            target_distribution=target_distribution,
            missing_values=missing_values,
            infinite_values=infinite_values,
            duplicate_rows=duplicate_rows,
            duplicate_columns=duplicate_columns,
            constant_features=constant_features,
            near_constant_features=near_constant_features,
            invalid_feature_types=invalid_feature_types,
            correlation_matrix=correlation_matrix,
            highly_correlated_features=highly_correlated_features,
            numeric_statistics=numeric_statistics,
            feature_summary=feature_summary,
            warnings=warnings,
            recommendations=recommendations,
            is_valid=is_valid,
        )

        self.logger.info(
            "Feature validation complete. Valid=%s", str(report.is_valid)
        )
        return report

    def _find_invalid_feature_types(
        self, df: pd.DataFrame,
    ) -> Dict[str, str]:
        """Identify non-numeric feature columns."""
        invalid_columns = {
            column: dtype.name
            for column, dtype in df.dtypes.items()
            if not pd.api.types.is_numeric_dtype(dtype)
        }
        self.logger.debug("Invalid feature types: %s", invalid_columns)
        return invalid_columns

    def _detect_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Detect missing values in the feature dataset."""
        missing = df.isna().sum()
        result = {column: int(count) for column, count in missing.items() if count > 0}
        self.logger.debug("Missing values: %s", result)
        return result

    def _detect_infinite_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Detect infinite values in numeric features."""
        numeric_df = df.select_dtypes(include=[np.number])
        result: Dict[str, int] = {}

        for column in numeric_df.columns:
            numeric_series = pd.to_numeric(numeric_df[column], errors="coerce")
            infinite_count = int(np.isinf(numeric_series).sum())
            if infinite_count > 0:
                result[column] = infinite_count

        self.logger.debug("Infinite values: %s", result)
        return result

    def _detect_duplicate_rows(self, df: pd.DataFrame) -> int:
        """Detect duplicate rows in the dataset."""
        duplicates = int(df.duplicated().sum())
        self.logger.debug("Duplicate rows: %d", duplicates)
        return duplicates

    def _detect_duplicate_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect duplicate columns with identical values."""
        duplicated_mask = df.T.duplicated(keep=False)
        duplicate_columns = [
            column for column, duplicate in zip(df.columns, duplicated_mask) if duplicate
        ]
        self.logger.debug("Duplicate columns: %s", duplicate_columns)
        return duplicate_columns

    def _detect_constant_features(self, df: pd.DataFrame) -> List[str]:
        """Detect features with a single unique value."""
        constant_features = [
            column
            for column in df.select_dtypes(include=[np.number]).columns
            if df[column].nunique(dropna=False) <= 1
        ]
        self.logger.debug("Constant features: %s", constant_features)
        return constant_features

    def _detect_near_constant_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Detect near-constant numeric features using unique value ratio."""
        near_constant: Dict[str, float] = {}
        numeric_df = df.select_dtypes(include=[np.number])
        total_rows = len(df)

        for column in numeric_df.columns:
            unique_ratio = numeric_df[column].nunique(dropna=False) / max(total_rows, 1)
            if 0 < unique_ratio <= self.near_constant_threshold:
                if numeric_df[column].nunique(dropna=False) > 1:
                    near_constant[column] = float(round(unique_ratio, 6))

        self.logger.debug("Near-constant features: %s", near_constant)
        return near_constant

    def _detect_target_distribution(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
    ) -> Dict[str, Any]:
        """Compute target distribution if a target column is provided."""
        if target_column is None:
            return {}

        if target_column not in df.columns:
            raise FeatureValidationError(
                f"Target column '{target_column}' does not exist in dataset."
            )

        distribution = self.statistics.target_distribution(df, target_column)
        self.logger.debug("Target distribution: %s", distribution)
        return distribution

    def _analyze_target_distribution(
        self,
        distribution: Dict[str, Any],
        warnings: List[str],
    ) -> None:
        """Analyze the target distribution for imbalances."""
        ratios = distribution.get("ratios", {})
        if not ratios:
            return

        imbalanced = {k: v for k, v in ratios.items() if float(v) >= 0.9}
        if imbalanced:
            warnings.append(
                f"Target distribution is highly imbalanced: {imbalanced}."
            )
            warnings.append(
                "Consider resampling or choosing a balanced target metric."
            )
