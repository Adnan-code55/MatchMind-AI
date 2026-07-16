"""
Data Splitting Engine.

This module provides a DatasetSplitter for splitting pandas DataFrames
into train, validation, and test sets with optional stratification and shuffling.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.data.logger import PipelineLogger
from .exceptions import DatasetTooSmallError, InvalidSplitRatioError, StratificationError
from .metadata import SplitMetadata


class DatasetSplitter:
    """Configurable dataset splitter for machine learning tasks."""

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.0,
        random_state: int = 42,
        shuffle: bool = True,
        stratify_column: Optional[str] = None,
    ) -> None:
        """Initialize the DatasetSplitter.

        Args:
            test_size: Proportion of the dataset to include in the test split.
            val_size: Proportion of the dataset to include in the validation split.
            random_state: Seed used by the random number generator.
            shuffle: Whether or not to shuffle the data before splitting.
            stratify_column: If not None, data is split in a stratified fashion, using this as the class labels.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify_column = stratify_column
        
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)
        self._validate_ratios()

    def _validate_ratios(self) -> None:
        """Validate split ratios.
        
        Raises:
            InvalidSplitRatioError: If ratios are out of bounds or sum >= 1.0.
        """
        if self.test_size < 0 or self.test_size >= 1.0:
            raise InvalidSplitRatioError(f"test_size must be between 0 and 1, got {self.test_size}")
            
        if self.val_size < 0 or self.val_size >= 1.0:
            raise InvalidSplitRatioError(f"val_size must be between 0 and 1, got {self.val_size}")
            
        total_split = self.test_size + self.val_size
        if total_split <= 0 or total_split >= 1.0:
            raise InvalidSplitRatioError(f"Sum of test_size and val_size must be strictly between 0 and 1, got {total_split}")

    def _validate_dataset(self, df: pd.DataFrame) -> None:
        """Validate input dataset.
        
        Args:
            df: The dataset to validate.
            
        Raises:
            DatasetTooSmallError: If the dataset has less than 3 samples.
            StratificationError: If stratify_column is provided but missing from DataFrame, or shuffle is False.
        """
        if len(df) < 3:
            raise DatasetTooSmallError(f"Dataset too small to split. Minimum required is 3, got {len(df)}.")

        if self.stratify_column is not None:
            if not self.shuffle:
                raise StratificationError("Stratified split requires shuffle=True.")
            if self.stratify_column not in df.columns:
                raise StratificationError(f"Stratification column '{self.stratify_column}' not found in dataset.")

    def split(
        self, df: pd.DataFrame
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, SplitMetadata], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata]]:
        """Split the dataset according to configuration.

        Args:
            df: The input pandas DataFrame.

        Returns:
            Tuple containing:
            - (train_df, test_df, metadata) if val_size is 0.0
            - (train_df, val_df, test_df, metadata) if val_size > 0.0
            
        Raises:
            DatasetTooSmallError: If dataset doesn't have enough samples.
            StratificationError: If stratification constraints are violated.
        """
        self.logger.info(f"Starting dataset split. Total rows: {len(df)}")
        self._validate_dataset(df)

        stratify_data = df[self.stratify_column] if self.stratify_column else None
        
        # Calculate intermediate test size for first split
        # We need to extract the validation set first if it exists, or just do a single split
        if self.val_size > 0:
            # First split: train+test vs val
            # Size of val relative to whole dataset
            train_test_df, val_df = train_test_split(
                df,
                test_size=self.val_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify_data
            )
            
            # Second split: train vs test
            # We need to stratify again using the remaining data
            stratify_data_remaining = train_test_df[self.stratify_column] if self.stratify_column else None
            
            # The new test size is relative to the remaining data
            # adjusted_test_size = original_test_size / (1.0 - original_val_size)
            adjusted_test_size = self.test_size / (1.0 - self.val_size)
            
            train_df, test_df = train_test_split(
                train_test_df,
                test_size=adjusted_test_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify_data_remaining
            )
        else:
            train_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify_data
            )
            val_df = pd.DataFrame()

        # Build metadata
        split_ratios = {"test": self.test_size}
        if self.val_size > 0:
            split_ratios["val"] = self.val_size

        metadata = SplitMetadata(
            train_count=len(train_df),
            test_count=len(test_df),
            val_count=len(val_df) if not val_df.empty else 0,
            split_ratios=split_ratios,
            random_seed=self.random_state,
            feature_count=len(df.columns),
            target_column=self.stratify_column
        )
        
        self.logger.info(f"Split successful. {metadata.describe()}")

        if self.val_size > 0:
            return train_df, val_df, test_df, metadata
        
        return train_df, test_df, metadata
