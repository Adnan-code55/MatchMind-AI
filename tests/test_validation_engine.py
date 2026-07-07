"""
Tests for the MatchMind AI feature validation engine.

These tests validate missing and infinite values, duplicate rows and columns,
constant and near-constant features, invalid feature types, and target
distribution analysis.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from backend.app.validation.feature_validator import FeatureValidator, FeatureValidationError
from backend.app.validation.statistics import ValidationStatistics
from backend.app.validation.dataset_report import DatasetValidationReport


def test_validate_missing_values() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, np.nan, 3.0],
            "feature_b": [0.0, 1.0, 2.0],
        }
    )

    validator = FeatureValidator()
    report = validator.validate(df)

    assert report.missing_values == {"feature_a": 1}
    assert report.is_valid is False
    assert "missing values" in report.warnings[0].lower()


def test_validate_infinite_values() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, np.inf, 3.0],
            "feature_b": [0.0, 1.0, -np.inf],
        }
    )

    validator = FeatureValidator()
    report = validator.validate(df)

    assert report.infinite_values == {"feature_a": 1, "feature_b": 1}
    assert report.is_valid is False
    assert any("infinite" in message.lower() for message in report.warnings)


def test_validate_duplicate_rows() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": [0.0, 1.0],
        }
    )
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    validator = FeatureValidator()
    report = validator.validate(df)

    assert report.duplicate_rows == 1
    assert report.is_valid is False
    assert any("duplicate row" in message.lower() for message in report.warnings)


def test_validate_duplicate_columns() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [1.0, 2.0, 3.0],
            "feature_c": [3.0, 4.0, 5.0],
        }
    )

    validator = FeatureValidator()
    report = validator.validate(df)

    assert set(report.duplicate_columns) == {"feature_a", "feature_b"}
    assert report.is_valid is False


def test_validate_constant_and_near_constant_features() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 1.0, 1.0, 1.0],
            "feature_b": [0.0, 0.0, 0.0, 1.0],
            "feature_c": [1.0, 2.0, 3.0, 4.0],
        }
    )

    validator = FeatureValidator(near_constant_threshold=0.5)
    report = validator.validate(df)

    assert report.constant_features == ["feature_a"]
    assert report.near_constant_features == {"feature_b": 0.5}
    assert report.is_valid is False


def test_validate_invalid_feature_types() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": ["x", "y", "z"],
        }
    )

    validator = FeatureValidator()
    report = validator.validate(df)

    assert report.invalid_feature_types["feature_b"] in {"str", "object", "string"}
    assert report.is_valid is False
    assert any("invalid feature types" in message.lower() for message in report.warnings)


def test_validate_target_distribution_warning() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )

    validator = FeatureValidator()
    report = validator.validate(df, target_column="target")

    assert "target distribution is highly imbalanced" in " ".join(report.warnings).lower()
    assert report.target_distribution["counts"] == {0: 9, 1: 1}
    assert report.is_valid is True


def test_numeric_statistics_and_correlation_summary() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [2.0, 4.0, 6.0],
            "feature_c": [3.0, 2.0, 1.0],
        }
    )

    validator = FeatureValidator(correlation_threshold=0.8)
    report = validator.validate(df)

    assert report.numeric_statistics["feature_a"]["mean"] == 2.0
    assert report.correlation_matrix["feature_a"]["feature_b"] == 1.0
    assert any(
        pair["feature_a"] == "feature_a" and pair["feature_b"] == "feature_b"
        for pair in report.highly_correlated_features
    )


def test_report_to_dict_serialization() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [2.0, 4.0, 6.0],
        }
    )

    validator = FeatureValidator()
    report = validator.validate(df)
    report_dict = report.to_dict()

    assert report_dict["rows"] == 3
    assert report_dict["columns"] == 2
    assert report_dict["feature_count"] == 2
    assert isinstance(report_dict["warnings"], list)
    assert isinstance(report_dict["recommendations"], list)


def test_error_for_invalid_target_column() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [2.0, 4.0, 6.0],
        }
    )

    validator = FeatureValidator()
    try:
        validator.validate(df, target_column="missing_target")
        assert False, "Expected FeatureValidationError for missing target column"
    except FeatureValidationError as exc:
        assert "target column" in str(exc).lower()


def test_stats_cache_reuse() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [2.0, 4.0, 6.0],
        }
    )
    cache = ValidationStatistics().cache
    validator = FeatureValidator(statistics=ValidationStatistics(cache=cache))
    first = validator.validate(df)
    second = validator.validate(df)

    assert first.correlation_matrix == second.correlation_matrix
    assert first.numeric_statistics == second.numeric_statistics
