"""
Tests for Data Splitting Engine.

Covers normal splits, reproducibility, ratio validation, invalid datasets,
stratified split, metadata generation, and exception handling.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.ml.splitting import (
    DatasetSplitter,
    DatasetTooSmallError,
    InvalidSplitRatioError,
    SplitMetadata,
    StratificationError,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample dataset for testing splits."""
    return pd.DataFrame({
        "Feature1": range(100),
        "Feature2": [float(i) for i in range(100)],
        "Target": ["A"] * 50 + ["B"] * 50,
    })


@pytest.fixture
def tiny_data() -> pd.DataFrame:
    """Create a dataset too small for splitting."""
    return pd.DataFrame({
        "Feature1": [1, 2],
        "Target": ["A", "B"]
    })


class TestDatasetSplitter:
    """Tests for the DatasetSplitter class."""

    def test_train_test_split_sizes(self, sample_data: pd.DataFrame) -> None:
        """Test basic train/test split sizes."""
        splitter = DatasetSplitter(test_size=0.2, val_size=0.0)
        train_df, test_df, metadata = splitter.split(sample_data)
        
        assert len(train_df) == 80
        assert len(test_df) == 20
        assert metadata.train_count == 80
        assert metadata.test_count == 20
        assert metadata.val_count == 0

    def test_train_val_test_split_sizes(self, sample_data: pd.DataFrame) -> None:
        """Test train/val/test split sizes."""
        splitter = DatasetSplitter(test_size=0.2, val_size=0.1)
        train_df, val_df, test_df, metadata = splitter.split(sample_data)
        
        # 10% of 100 is 10, 20% of 100 is 20, remaining is 70
        assert len(train_df) == 70
        assert len(val_df) == 10
        assert len(test_df) == 20
        
        assert metadata.train_count == 70
        assert metadata.val_count == 10
        assert metadata.test_count == 20

    def test_reproducibility(self, sample_data: pd.DataFrame) -> None:
        """Test that splits are reproducible with same random_state."""
        splitter_1 = DatasetSplitter(test_size=0.2, random_state=42)
        train_1, test_1, _ = splitter_1.split(sample_data)
        
        splitter_2 = DatasetSplitter(test_size=0.2, random_state=42)
        train_2, test_2, _ = splitter_2.split(sample_data)
        
        pd.testing.assert_frame_equal(train_1, train_2)
        pd.testing.assert_frame_equal(test_1, test_2)

    def test_invalid_ratio_validation(self) -> None:
        """Test validation of invalid split ratios."""
        with pytest.raises(InvalidSplitRatioError):
            DatasetSplitter(test_size=1.2)
            
        with pytest.raises(InvalidSplitRatioError):
            DatasetSplitter(test_size=-0.1)
            
        with pytest.raises(InvalidSplitRatioError):
            DatasetSplitter(test_size=0.6, val_size=0.5)

    def test_dataset_too_small(self, tiny_data: pd.DataFrame) -> None:
        """Test validation of minimum dataset size."""
        splitter = DatasetSplitter(test_size=0.5)
        with pytest.raises(DatasetTooSmallError):
            splitter.split(tiny_data)

    def test_stratified_split(self, sample_data: pd.DataFrame) -> None:
        """Test stratified splitting maintains class ratios."""
        splitter = DatasetSplitter(test_size=0.2, stratify_column="Target")
        train_df, test_df, _ = splitter.split(sample_data)
        
        # Train should have 40 'A' and 40 'B' (total 80)
        assert train_df["Target"].value_counts()["A"] == 40
        assert train_df["Target"].value_counts()["B"] == 40
        
        # Test should have 10 'A' and 10 'B' (total 20)
        assert test_df["Target"].value_counts()["A"] == 10
        assert test_df["Target"].value_counts()["B"] == 10

    def test_stratification_missing_column(self, sample_data: pd.DataFrame) -> None:
        """Test stratification fails if column is missing."""
        splitter = DatasetSplitter(test_size=0.2, stratify_column="MissingTarget")
        with pytest.raises(StratificationError):
            splitter.split(sample_data)
            
    def test_stratification_no_shuffle(self, sample_data: pd.DataFrame) -> None:
        """Test stratification fails if shuffle is False."""
        splitter = DatasetSplitter(test_size=0.2, shuffle=False, stratify_column="Target")
        with pytest.raises(StratificationError):
            splitter.split(sample_data)

    def test_metadata_generation(self, sample_data: pd.DataFrame) -> None:
        """Test SplitMetadata accurately represents the split."""
        splitter = DatasetSplitter(test_size=0.3, val_size=0.1, stratify_column="Target", random_state=99)
        _, _, _, metadata = splitter.split(sample_data)
        
        assert isinstance(metadata, SplitMetadata)
        assert metadata.train_count == 60
        assert metadata.test_count == 30
        assert metadata.val_count == 10
        assert metadata.split_ratios == {"test": 0.3, "val": 0.1}
        assert metadata.random_seed == 99
        assert metadata.feature_count == 3
        assert metadata.target_column == "Target"
        assert metadata.timestamp is not None
        
        # Check serialization methods
        metadata_dict = metadata.to_dict()
        assert "train_count" in metadata_dict
        assert "timestamp" in metadata_dict
        
        description = metadata.describe()
        assert "Val=10" in description
        assert "Target" in description
