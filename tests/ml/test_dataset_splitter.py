"""
Tests for chronological dataset splitter.

This module contains comprehensive tests for the ChronologicalDatasetSplitter
to ensure correct behavior, data integrity, and chronological ordering.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from backend.ml import (
    ChronologicalDatasetSplitter,
    DatasetTooSmallError,
    DuplicateMatchError,
    InvalidSplitConfiguration,
    MissingDateColumnError,
    SplitConfig,
)


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """Create sample football match data for testing.

    Returns:
        DataFrame with 20 sample matches spanning 20 days.
    """
    dates = [
        datetime(2023, 1, 1) + timedelta(days=i) for i in range(20)
    ]
    matches = []
    teams = ["Arsenal", "Chelsea", "Liverpool", "Manchester United"]

    for i, date in enumerate(dates):
        matches.append(
            {
                "Date": date,
                "HomeTeam": teams[i % len(teams)],
                "AwayTeam": teams[(i + 1) % len(teams)],
                "FTHG": i % 4,
                "FTAG": (i + 1) % 4,
                "FTR": ["H", "D", "A"][i % 3],
            }
        )

    return pd.DataFrame(matches)


class TestSplitConfig:
    """Tests for SplitConfig validation."""

    def test_default_config_is_valid(self) -> None:
        """Test that default configuration is valid."""
        config = SplitConfig()
        assert config.train_ratio == 0.70
        assert config.validation_ratio == 0.15
        assert config.test_ratio == 0.15
        assert config.shuffle is False

    def test_custom_valid_config(self) -> None:
        """Test that custom valid ratios are accepted."""
        config = SplitConfig(train_ratio=0.80, validation_ratio=0.10, test_ratio=0.10)
        assert config.train_ratio == 0.80

    def test_ratios_must_sum_to_one(self) -> None:
        """Test that ratios must sum to 1.0."""
        with pytest.raises(InvalidSplitConfiguration):
            SplitConfig(train_ratio=0.70, validation_ratio=0.20, test_ratio=0.20)

    def test_train_ratio_must_be_positive(self) -> None:
        """Test that train ratio must be positive."""
        with pytest.raises(InvalidSplitConfiguration):
            SplitConfig(train_ratio=0.0, validation_ratio=0.5, test_ratio=0.5)

    def test_validation_ratio_can_be_zero(self) -> None:
        """Test that validation ratio can be zero."""
        config = SplitConfig(train_ratio=0.80, validation_ratio=0.0, test_ratio=0.20)
        assert config.validation_ratio == 0.0

    def test_test_ratio_can_be_zero(self) -> None:
        """Test that test ratio can be zero."""
        config = SplitConfig(train_ratio=0.80, validation_ratio=0.20, test_ratio=0.0)
        assert config.test_ratio == 0.0

    def test_describe_method(self) -> None:
        """Test configuration description string."""
        config = SplitConfig()
        description = config.describe()
        assert "70.00%" in description
        assert "15.00%" in description
        assert "shuffle=False" in description


class TestChronologicalDatasetSplitter:
    """Tests for the ChronologicalDatasetSplitter."""

    def test_normal_split_with_default_config(self, sample_matches: pd.DataFrame) -> None:
        """Test normal split produces three non-empty sets."""
        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(sample_matches)

        assert len(result.train_df) == 14  # 70% of 20
        assert len(result.validation_df) == 3  # 15% of 20
        assert len(result.test_df) == 3  # 15% of 20

    def test_split_preserves_chronological_order(self, sample_matches: pd.DataFrame) -> None:
        """Test that chronological order is preserved within splits."""
        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(sample_matches)

        # Check train set is in chronological order
        train_dates = pd.to_datetime(result.train_df["Date"])
        assert (train_dates.diff().dropna() >= pd.Timedelta(0)).all()

        # Check validation set is in chronological order
        val_dates = pd.to_datetime(result.validation_df["Date"])
        assert (val_dates.diff().dropna() >= pd.Timedelta(0)).all()

        # Check test set is in chronological order
        test_dates = pd.to_datetime(result.test_df["Date"])
        assert (test_dates.diff().dropna() >= pd.Timedelta(0)).all()

    def test_split_order_is_temporal(self, sample_matches: pd.DataFrame) -> None:
        """Test that train comes before validation, validation before test."""
        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(sample_matches)

        if len(result.train_df) > 0 and len(result.validation_df) > 0:
            train_max = pd.to_datetime(result.train_df["Date"]).max()
            val_min = pd.to_datetime(result.validation_df["Date"]).min()
            assert train_max <= val_min

        if len(result.validation_df) > 0 and len(result.test_df) > 0:
            val_max = pd.to_datetime(result.validation_df["Date"]).max()
            test_min = pd.to_datetime(result.test_df["Date"]).min()
            assert val_max <= test_min

    def test_split_ratios_approximate_config(self, sample_matches: pd.DataFrame) -> None:
        """Test that split ratios approximate the configured values."""
        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(sample_matches)

        total = len(sample_matches)
        train_ratio = len(result.train_df) / total
        val_ratio = len(result.validation_df) / total
        test_ratio = len(result.test_df) / total

        # Allow 5% tolerance due to integer rounding
        assert abs(train_ratio - 0.70) < 0.05
        assert abs(val_ratio - 0.15) < 0.05
        assert abs(test_ratio - 0.15) < 0.05

    def test_custom_split_config(self, sample_matches: pd.DataFrame) -> None:
        """Test split with custom configuration."""
        config = SplitConfig(train_ratio=0.60, validation_ratio=0.20, test_ratio=0.20)
        splitter = ChronologicalDatasetSplitter(config)
        result = splitter.split(sample_matches)

        assert len(result.train_df) == 12  # 60% of 20
        assert len(result.validation_df) == 4  # 20% of 20
        assert len(result.test_df) == 4  # 20% of 20

    def test_split_with_no_validation_set(self, sample_matches: pd.DataFrame) -> None:
        """Test split with no validation set."""
        config = SplitConfig(train_ratio=0.80, validation_ratio=0.0, test_ratio=0.20)
        splitter = ChronologicalDatasetSplitter(config)
        result = splitter.split(sample_matches)

        assert len(result.train_df) == 16  # 80% of 20
        assert len(result.validation_df) == 0  # 0% of 20
        assert len(result.test_df) == 4  # 20% of 20

    def test_split_with_no_test_set(self, sample_matches: pd.DataFrame) -> None:
        """Test split with no test set."""
        config = SplitConfig(train_ratio=0.80, validation_ratio=0.20, test_ratio=0.0)
        splitter = ChronologicalDatasetSplitter(config)
        result = splitter.split(sample_matches)

        assert len(result.train_df) == 16  # 80% of 20
        assert len(result.validation_df) == 4  # 20% of 20
        assert len(result.test_df) == 0  # 0% of 20

    def test_very_small_dataset_rejected(self) -> None:
        """Test that datasets with fewer than 3 rows are rejected."""
        df = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "HomeTeam": ["A", "B"],
            "AwayTeam": ["C", "D"],
            "FTHG": [1, 0],
            "FTAG": [0, 1],
        })

        splitter = ChronologicalDatasetSplitter()
        with pytest.raises(DatasetTooSmallError):
            splitter.split(df)

    def test_empty_dataset_rejected(self) -> None:
        """Test that empty datasets are rejected."""
        df = pd.DataFrame({
            "Date": [],
            "HomeTeam": [],
            "AwayTeam": [],
            "FTHG": [],
            "FTAG": [],
        })

        splitter = ChronologicalDatasetSplitter()
        with pytest.raises(DatasetTooSmallError):
            splitter.split(df)

    def test_missing_date_column_rejected(self) -> None:
        """Test that datasets without Date column are rejected."""
        df = pd.DataFrame({
            "HomeTeam": ["A", "B", "C"],
            "AwayTeam": ["D", "E", "F"],
            "FTHG": [1, 0, 2],
            "FTAG": [0, 1, 0],
        })

        splitter = ChronologicalDatasetSplitter()
        with pytest.raises(MissingDateColumnError):
            splitter.split(df)

    def test_duplicate_match_records_detected(self) -> None:
        """Test that exact duplicate match records are detected."""
        df = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-01", "2023-01-02"],
            "HomeTeam": ["A", "A", "B"],
            "AwayTeam": ["B", "B", "C"],
            "FTHG": [1, 1, 2],
            "FTAG": [0, 0, 1],
        })

        splitter = ChronologicalDatasetSplitter()
        with pytest.raises(DuplicateMatchError):
            splitter.split(df)

    def test_multiple_matches_same_date_allowed(self, sample_matches: pd.DataFrame) -> None:
        """Test that multiple different matches on same date are allowed."""
        # Create a dataset with multiple matches on the same date
        df = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-01", "2023-01-02"],
            "HomeTeam": ["A", "B", "C"],
            "AwayTeam": ["B", "C", "D"],
            "FTHG": [1, 2, 0],
            "FTAG": [0, 1, 3],
        })

        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(df)

        # Should not raise and should produce valid split
        assert len(result.train_df) + len(result.validation_df) + len(result.test_df) == 3

    def test_original_dataframe_not_modified(self, sample_matches: pd.DataFrame) -> None:
        """Test that the original DataFrame is not modified during split."""
        original_copy = sample_matches.copy()
        splitter = ChronologicalDatasetSplitter()
        splitter.split(sample_matches)

        # Verify original is unchanged
        pd.testing.assert_frame_equal(sample_matches, original_copy)

    def test_splits_are_independent_copies(self, sample_matches: pd.DataFrame) -> None:
        """Test that split DataFrames are independent copies."""
        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(sample_matches)

        # Modify a split
        result.train_df.loc[0, "HomeTeam"] = "Modified"

        # Verify original is unchanged
        assert sample_matches.loc[0, "HomeTeam"] != "Modified"

    def test_no_overlap_between_splits(self, sample_matches: pd.DataFrame) -> None:
        """Test that no indices overlap between splits."""
        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(sample_matches)

        train_indices = set(result.train_df.index)
        val_indices = set(result.validation_df.index)
        test_indices = set(result.test_df.index)

        assert not (train_indices & val_indices)
        assert not (train_indices & test_indices)
        assert not (val_indices & test_indices)

    def test_complete_coverage(self, sample_matches: pd.DataFrame) -> None:
        """Test that all rows are accounted for in splits."""
        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(sample_matches)

        total_split_rows = (
            len(result.train_df)
            + len(result.validation_df)
            + len(result.test_df)
        )

        assert total_split_rows == len(sample_matches)

    def test_metadata_correctness(self, sample_matches: pd.DataFrame) -> None:
        """Test that metadata accurately reflects the split."""
        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(sample_matches)

        assert result.metadata.total_rows == 20
        assert result.metadata.train_rows == 14
        assert result.metadata.validation_rows == 3
        assert result.metadata.test_rows == 3

    def test_metadata_date_range(self, sample_matches: pd.DataFrame) -> None:
        """Test that metadata contains correct date range."""
        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(sample_matches)

        min_date = pd.to_datetime(sample_matches["Date"]).min()
        max_date = pd.to_datetime(sample_matches["Date"]).max()

        assert result.metadata.date_range[0] == min_date
        assert result.metadata.date_range[1] == max_date

    def test_metadata_has_timestamp(self, sample_matches: pd.DataFrame) -> None:
        """Test that metadata includes a timestamp."""
        before = datetime.now()
        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(sample_matches)
        after = datetime.now()

        assert before <= result.metadata.timestamp <= after

    def test_split_result_to_dict(self, sample_matches: pd.DataFrame) -> None:
        """Test metadata serialization to dictionary."""
        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(sample_matches)

        metadata_dict = result.metadata.to_dict()

        assert metadata_dict["total_rows"] == 20
        assert metadata_dict["train_rows"] == 14
        assert isinstance(metadata_dict["date_range"], tuple)
        assert len(metadata_dict["date_range"]) == 2

    def test_splitter_describe(self) -> None:
        """Test splitter description."""
        splitter = ChronologicalDatasetSplitter()
        description = splitter.describe()

        assert "ChronologicalDatasetSplitter" in description
        assert "70.00%" in description

    def test_metadata_describe(self, sample_matches: pd.DataFrame) -> None:
        """Test metadata description."""
        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(sample_matches)

        description = result.metadata.describe()

        assert "20 rows" in description
        assert "Train:" in description
        assert "Validation:" in description
        assert "Test:" in description

    def test_split_with_single_day_multiple_matches(self) -> None:
        """Test split with many matches on a single day."""
        # Create unique teams for each match to avoid duplicate detection
        home_teams = ["Team" + str(i % 10) for i in range(20)]
        away_teams = ["Team" + str((i + 1) % 10) for i in range(20)]
        
        df = pd.DataFrame({
            "Date": ["2023-01-01"] * 15 + ["2023-01-02"] * 5,
            "HomeTeam": home_teams,
            "AwayTeam": away_teams,
            "FTHG": list(range(20)),
            "FTAG": list(range(1, 21)),  # Offset to ensure different results
        })

        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(df)

        assert len(result.train_df) + len(result.validation_df) + len(result.test_df) == 20

    def test_split_handles_unsorted_input(self) -> None:
        """Test that split correctly sorts unsorted input."""
        df = pd.DataFrame({
            "Date": ["2023-01-03", "2023-01-01", "2023-01-02", "2023-01-04"],
            "HomeTeam": ["A", "B", "C", "D"],
            "AwayTeam": ["E", "F", "G", "H"],
            "FTHG": [1, 2, 3, 0],
            "FTAG": [0, 1, 0, 2],
        })

        splitter = ChronologicalDatasetSplitter()
        result = splitter.split(df)

        # Verify train dates are sorted
        train_dates = pd.to_datetime(result.train_df["Date"]).tolist()
        assert train_dates == sorted(train_dates)


class TestSplitResultVerification:
    """Tests for SplitResult verification methods."""

    def test_verify_no_overlap_passes(self) -> None:
        """Test that verify_no_overlap passes for non-overlapping sets."""
        train_df = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])
        val_df = pd.DataFrame({"A": [4, 5]}, index=[3, 4])
        test_df = pd.DataFrame({"A": [6, 7]}, index=[5, 6])

        from backend.ml import SplitMetadata, SplitResult

        result = SplitResult(
            train_df=train_df,
            validation_df=val_df,
            test_df=test_df,
            metadata=SplitMetadata(
                total_rows=7,
                train_rows=3,
                validation_rows=2,
                test_rows=2,
                date_range=(datetime.now(), datetime.now()),
                train_date_boundary=datetime.now(),
                validation_date_boundary=datetime.now(),
                test_date_boundary=datetime.now(),
            ),
        )

        assert result.verify_no_overlap() is True

    def test_verify_complete_coverage_passes(self) -> None:
        """Test that verify_complete_coverage passes for complete splits."""
        train_df = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])
        val_df = pd.DataFrame({"A": [4, 5]}, index=[3, 4])
        test_df = pd.DataFrame({"A": [6, 7]}, index=[5, 6])

        from backend.ml import SplitMetadata, SplitResult

        result = SplitResult(
            train_df=train_df,
            validation_df=val_df,
            test_df=test_df,
            metadata=SplitMetadata(
                total_rows=7,
                train_rows=3,
                validation_rows=2,
                test_rows=2,
                date_range=(datetime.now(), datetime.now()),
                train_date_boundary=datetime.now(),
                validation_date_boundary=datetime.now(),
                test_date_boundary=datetime.now(),
            ),
        )

        assert result.verify_complete_coverage(7) is True
