"""
Tests for preprocessing pipeline.

Comprehensive test suite covering all preprocessing functionality
including transformers, feature detection, and metadata generation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.ml.preprocessing import (
    CategoricalEncoder,
    FeatureDetector,
    FeatureScaler,
    MissingValueHandler,
    PreprocessingError,
    PreprocessingPipeline,
    TargetExtractor,
    TransformationStateError,
    UnsupportedTransformationError,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample dataset for testing.

    Returns:
        DataFrame with mixed feature types.
    """
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=20),
        "HomeTeam": ["Team" + str(i % 5) for i in range(20)],
        "AwayTeam": ["Team" + str((i + 1) % 5) for i in range(20)],
        "Attendance": [40000 + i * 100 for i in range(20)],
        "HomeGoals": [1, 2, None, 0, 3, 1, 2, 1, 0, 2, 1, 2, 1, 0, 2, 1, 2, 1, 0, 2],
        "AwayGoals": [0, 1, 2, None, 1, 0, 1, 2, 3, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1],
        "Result": ["H", "H", "A", "D", "H", "H", "H", "A", "A", "H", "H", "H", "A", "D", "H", "H", "H", "A", "A", "H"],
    })


class TestFeatureDetector:
    """Tests for FeatureDetector."""

    def test_detect_features_correctly_classifies_types(self, sample_data: pd.DataFrame) -> None:
        """Test that feature types are correctly detected."""
        detector = FeatureDetector()
        features = detector.detect_features(sample_data)

        assert "Date" in features["date"]
        assert "HomeTeam" in features["categorical"]
        assert "AwayTeam" in features["categorical"]
        assert "Attendance" in features["numerical"]
        assert "HomeGoals" in features["numerical"]
        assert "AwayGoals" in features["numerical"]
        assert "Result" in features["categorical"]

    def test_detect_features_excludes_target_column(self, sample_data: pd.DataFrame) -> None:
        """Test that target column is excluded from detection."""
        detector = FeatureDetector()
        features = detector.detect_features(sample_data, target_column="Result")

        # Result should not appear in feature lists
        assert "Result" not in features["categorical"]


class TestMissingValueHandler:
    """Tests for MissingValueHandler."""

    def test_mean_strategy_fills_numerical_missing(self, sample_data: pd.DataFrame) -> None:
        """Test that mean strategy fills numerical missing values."""
        handler = MissingValueHandler(numerical_strategy="mean")
        handler.fit(sample_data, numerical_cols=["HomeGoals", "AwayGoals"])
        result = handler.transform(sample_data)

        assert not result["HomeGoals"].isna().any()
        assert not result["AwayGoals"].isna().any()

    def test_median_strategy_fills_numerical_missing(self, sample_data: pd.DataFrame) -> None:
        """Test that median strategy fills numerical missing values."""
        handler = MissingValueHandler(numerical_strategy="median")
        handler.fit(sample_data, numerical_cols=["HomeGoals"])
        result = handler.transform(sample_data)

        assert not result["HomeGoals"].isna().any()

    def test_constant_strategy_fills_numerical_missing(self, sample_data: pd.DataFrame) -> None:
        """Test that constant strategy fills numerical missing values."""
        handler = MissingValueHandler(
            numerical_strategy="constant",
            numerical_constant=0.0
        )
        handler.fit(sample_data, numerical_cols=["HomeGoals"])
        result = handler.transform(sample_data)

        assert not result["HomeGoals"].isna().any()
        assert (result[result["HomeGoals"] == 0.0]).shape[0] > 0

    def test_most_frequent_strategy_fills_categorical_missing(self, sample_data: pd.DataFrame) -> None:
        """Test that most_frequent strategy fills categorical missing values."""
        df = sample_data.copy()
        df.loc[0, "HomeTeam"] = None
        df.loc[1, "HomeTeam"] = None

        handler = MissingValueHandler(categorical_strategy="most_frequent")
        handler.fit(df, categorical_cols=["HomeTeam"])
        result = handler.transform(df)

        assert not result["HomeTeam"].isna().any()

    def test_invalid_numerical_strategy_raises_error(self) -> None:
        """Test that invalid numerical strategy raises error."""
        with pytest.raises(UnsupportedTransformationError):
            MissingValueHandler(numerical_strategy="invalid")

    def test_invalid_categorical_strategy_raises_error(self) -> None:
        """Test that invalid categorical strategy raises error."""
        with pytest.raises(UnsupportedTransformationError):
            MissingValueHandler(categorical_strategy="invalid")

    def test_transform_before_fit_raises_error(self, sample_data: pd.DataFrame) -> None:
        """Test that transforming before fitting raises error."""
        handler = MissingValueHandler()

        with pytest.raises(TransformationStateError):
            handler.transform(sample_data)


class TestCategoricalEncoder:
    """Tests for CategoricalEncoder."""

    def test_label_encoding_on_low_cardinality(self, sample_data: pd.DataFrame) -> None:
        """Test label encoding on low-cardinality categorical."""
        encoder = CategoricalEncoder(encoding_type="label")
        encoder.fit(sample_data, categorical_cols=["Result"])
        result = encoder.transform(sample_data)

        # Result column should be numeric
        assert pd.api.types.is_numeric_dtype(result["Result"])

    def test_onehot_encoding_creates_new_columns(self, sample_data: pd.DataFrame) -> None:
        """Test that one-hot encoding creates new columns."""
        encoder = CategoricalEncoder(encoding_type="onehot")
        encoder.fit(sample_data, categorical_cols=["Result"])
        result = encoder.transform(sample_data)

        # Result column should be dropped
        assert "Result" not in result.columns

        # New columns should be created
        assert any("Result_" in col for col in result.columns)

    def test_auto_encoding_chooses_based_on_cardinality(self, sample_data: pd.DataFrame) -> None:
        """Test that auto encoding chooses strategy based on cardinality."""
        df = sample_data.copy()
        df["LowCardinality"] = ["A", "B"] * 10  # 2 unique values
        df["HighCardinality"] = range(20)  # 20 unique values

        encoder = CategoricalEncoder(encoding_type="auto", max_categories_label=10)
        encoder.fit(df, categorical_cols=["LowCardinality", "HighCardinality"])

        # Low cardinality should use label encoding
        assert encoder.encoding_strategy["LowCardinality"] == "label"

        # High cardinality should use one-hot
        assert encoder.encoding_strategy["HighCardinality"] == "onehot"

    def test_invalid_encoding_type_raises_error(self) -> None:
        """Test that invalid encoding type raises error."""
        with pytest.raises(UnsupportedTransformationError):
            CategoricalEncoder(encoding_type="invalid")

    def test_transform_before_fit_raises_error(self, sample_data: pd.DataFrame) -> None:
        """Test that transforming before fitting raises error."""
        encoder = CategoricalEncoder()

        with pytest.raises(TransformationStateError):
            encoder.transform(sample_data)


class TestFeatureScaler:
    """Tests for FeatureScaler."""

    def test_standard_scaler_scales_values(self, sample_data: pd.DataFrame) -> None:
        """Test that standard scaler scales numerical values."""
        scaler = FeatureScaler(scaler_type="standard", enabled=True)
        scaler.fit(sample_data, numerical_cols=["Attendance"])
        result = scaler.transform(sample_data)

        # Scaled values should have mean near 0 and std near 1
        assert abs(result["Attendance"].mean()) < 1
        assert abs(result["Attendance"].std() - 1) < 0.1

    def test_minmax_scaler_scales_to_0_1(self, sample_data: pd.DataFrame) -> None:
        """Test that minmax scaler scales values to [0, 1]."""
        scaler = FeatureScaler(scaler_type="minmax", enabled=True)
        scaler.fit(sample_data, numerical_cols=["Attendance"])
        result = scaler.transform(sample_data)

        # Scaled values should be in [0, 1]
        assert result["Attendance"].min() >= -0.01  # Small tolerance for float precision
        assert result["Attendance"].max() <= 1.01

    def test_scaling_disabled_returns_unchanged(self, sample_data: pd.DataFrame) -> None:
        """Test that disabled scaling returns unchanged data."""
        original_values = sample_data["Attendance"].copy()

        scaler = FeatureScaler(enabled=False)
        scaler.fit(sample_data, numerical_cols=["Attendance"])
        result = scaler.transform(sample_data)

        pd.testing.assert_series_equal(result["Attendance"], original_values)

    def test_invalid_scaler_type_raises_error(self) -> None:
        """Test that invalid scaler type raises error."""
        with pytest.raises(UnsupportedTransformationError):
            FeatureScaler(scaler_type="invalid")

    def test_transform_before_fit_raises_error(self, sample_data: pd.DataFrame) -> None:
        """Test that transforming before fitting raises error."""
        scaler = FeatureScaler()

        with pytest.raises(TransformationStateError):
            scaler.transform(sample_data)


class TestTargetExtractor:
    """Tests for TargetExtractor."""

    def test_extracts_target_column(self, sample_data: pd.DataFrame) -> None:
        """Test that target column is correctly extracted."""
        extractor = TargetExtractor()
        extractor.fit(sample_data, target_column="Result")
        features, target = extractor.transform(sample_data)

        assert "Result" not in features.columns
        assert target.name == "Result"
        assert len(target) == len(sample_data)

    def test_missing_target_column_raises_error(self, sample_data: pd.DataFrame) -> None:
        """Test that missing target column raises error."""
        extractor = TargetExtractor()

        with pytest.raises(ValueError):
            extractor.fit(sample_data, target_column="NonExistent")

    def test_transform_before_fit_raises_error(self, sample_data: pd.DataFrame) -> None:
        """Test that transforming before fitting raises error."""
        extractor = TargetExtractor()

        with pytest.raises(TransformationStateError):
            extractor.transform(sample_data)


class TestPreprocessingPipeline:
    """Tests for PreprocessingPipeline."""

    def test_pipeline_fit_transform_completes(self, sample_data: pd.DataFrame) -> None:
        """Test that pipeline fit_transform completes successfully."""
        pipeline = PreprocessingPipeline(
            target_column="Result",
            enable_scaling=True,
        )

        features, target = pipeline.fit_transform(sample_data)

        assert features is not None
        assert target is not None
        assert len(features) == len(sample_data)
        assert len(target) == len(sample_data)

    def test_pipeline_preserves_row_count(self, sample_data: pd.DataFrame) -> None:
        """Test that pipeline preserves number of rows."""
        pipeline = PreprocessingPipeline(target_column="Result")
        features, target = pipeline.fit_transform(sample_data)

        assert len(features) == len(sample_data)

    def test_pipeline_removes_missing_values(self, sample_data: pd.DataFrame) -> None:
        """Test that pipeline handles missing values."""
        pipeline = PreprocessingPipeline(
            target_column="Result",
            numerical_missing_strategy="mean",
        )

        features, target = pipeline.fit_transform(sample_data)

        # Check that features have no missing values (except metadata dropped)
        for col in features.columns:
            if col != "Date":  # Date might be dropped
                assert not features[col].isna().any()

    def test_pipeline_encodes_categorical_features(self, sample_data: pd.DataFrame) -> None:
        """Test that categorical features are encoded."""
        pipeline = PreprocessingPipeline(
            target_column="Result",
            encoding_type="label",
        )

        features, target = pipeline.fit_transform(sample_data)

        # HomeTeam and AwayTeam should be encoded (numeric)
        if "HomeTeam" in features.columns:
            assert pd.api.types.is_numeric_dtype(features["HomeTeam"])
        if "AwayTeam" in features.columns:
            assert pd.api.types.is_numeric_dtype(features["AwayTeam"])

    def test_pipeline_scales_features_when_enabled(self, sample_data: pd.DataFrame) -> None:
        """Test that features are scaled when enabled."""
        pipeline = PreprocessingPipeline(
            target_column="Result",
            enable_scaling=True,
            scaler_type="standard",
        )

        features, target = pipeline.fit_transform(sample_data)

        # Numerical features should be scaled
        if "Attendance" in features.columns:
            assert abs(features["Attendance"].std() - 1) < 1  # Rough check

    def test_pipeline_skips_scaling_when_disabled(self, sample_data: pd.DataFrame) -> None:
        """Test that scaling is skipped when disabled."""
        original_data = sample_data.copy()

        pipeline = PreprocessingPipeline(
            target_column="Result",
            enable_scaling=False,
        )

        features, target = pipeline.fit_transform(sample_data)

        # Note: values may still differ due to encoding/missing handling
        # So we just check that scaling transformers are created but disabled
        assert pipeline.feature_scaler.enabled is False

    def test_pipeline_drops_date_columns_by_default(self, sample_data: pd.DataFrame) -> None:
        """Test that date columns are dropped by default."""
        pipeline = PreprocessingPipeline(
            target_column="Result",
            drop_date_columns=True,
        )

        features, target = pipeline.fit_transform(sample_data)

        # Date column should be dropped
        assert "Date" not in features.columns

    def test_pipeline_preserves_date_columns_when_disabled(self, sample_data: pd.DataFrame) -> None:
        """Test that date columns are preserved when dropping disabled."""
        pipeline = PreprocessingPipeline(
            target_column="Result",
            drop_date_columns=False,
        )

        features, target = pipeline.fit_transform(sample_data)

        # Date column should still exist (but transformed to numeric or str)
        assert "Date" in features.columns or any("date" in col.lower() for col in features.columns)

    def test_pipeline_generates_metadata(self, sample_data: pd.DataFrame) -> None:
        """Test that metadata is generated correctly."""
        pipeline = PreprocessingPipeline(target_column="Result")
        pipeline.fit_transform(sample_data)

        metadata = pipeline.get_metadata()

        assert metadata is not None
        assert metadata.original_shape[0] == len(sample_data)
        assert metadata.target_column == "Result"
        assert len(metadata.features) > 0

    def test_pipeline_metadata_tracks_encoded_features(self, sample_data: pd.DataFrame) -> None:
        """Test that metadata tracks encoded features."""
        pipeline = PreprocessingPipeline(
            target_column="Result",
            encoding_type="label",
        )
        pipeline.fit_transform(sample_data)

        metadata = pipeline.get_metadata()

        assert len(metadata.encoded_features) > 0

    def test_pipeline_metadata_tracks_scaled_features(self, sample_data: pd.DataFrame) -> None:
        """Test that metadata tracks scaled features."""
        pipeline = PreprocessingPipeline(
            target_column="Result",
            enable_scaling=True,
        )
        pipeline.fit_transform(sample_data)

        metadata = pipeline.get_metadata()

        assert len(metadata.scaled_features) > 0

    def test_pipeline_metadata_tracks_dropped_features(self, sample_data: pd.DataFrame) -> None:
        """Test that metadata tracks dropped features."""
        pipeline = PreprocessingPipeline(
            target_column="Result",
            drop_date_columns=True,
        )
        pipeline.fit_transform(sample_data)

        metadata = pipeline.get_metadata()

        assert "Date" in metadata.dropped_features

    def test_pipeline_missing_target_raises_error(self, sample_data: pd.DataFrame) -> None:
        """Test that missing target column raises error."""
        pipeline = PreprocessingPipeline(target_column="NonExistent")

        with pytest.raises(Exception):
            pipeline.fit(sample_data)

    def test_pipeline_transform_before_fit_raises_error(self, sample_data: pd.DataFrame) -> None:
        """Test that transforming before fitting raises error."""
        pipeline = PreprocessingPipeline()

        with pytest.raises(ValueError):
            pipeline.transform(sample_data)

    def test_pipeline_original_dataframe_unchanged(self, sample_data: pd.DataFrame) -> None:
        """Test that original DataFrame is never modified."""
        original_copy = sample_data.copy()

        pipeline = PreprocessingPipeline(target_column="Result")
        pipeline.fit_transform(sample_data)

        pd.testing.assert_frame_equal(sample_data, original_copy)

    def test_pipeline_fit_transform_consistency(self, sample_data: pd.DataFrame) -> None:
        """Test that fit_transform produces same results as fit then transform."""
        pipeline1 = PreprocessingPipeline(target_column="Result")
        features1, target1 = pipeline1.fit_transform(sample_data)

        pipeline2 = PreprocessingPipeline(target_column="Result")
        pipeline2.fit(sample_data)
        features2, target2 = pipeline2.transform(sample_data)

        pd.testing.assert_frame_equal(features1, features2)
        pd.testing.assert_series_equal(target1, target2)

    def test_pipeline_describe_method(self) -> None:
        """Test pipeline description method."""
        pipeline = PreprocessingPipeline(
            target_column="Result",
            enable_scaling=True,
        )

        description = pipeline.describe()

        assert "PreprocessingPipeline" in description
        assert "Result" in description

    def test_pipeline_metadata_describe_method(self, sample_data: pd.DataFrame) -> None:
        """Test metadata description method."""
        pipeline = PreprocessingPipeline(target_column="Result")
        pipeline.fit_transform(sample_data)

        metadata = pipeline.get_metadata()
        description = metadata.describe()

        assert "rows" in description
        assert "Encoded" in description
        assert "Scaled" in description
