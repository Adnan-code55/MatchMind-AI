"""
Chronological dataset splitter for time-series football data.

This module implements train/validation/test splitting that preserves
chronological order to prevent data leakage in machine learning pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from backend.app.data.logger import PipelineLogger
from .exceptions import (
    DatasetTooSmallError,
    DuplicateMatchError,
    MissingDateColumnError,
)
from .split_config import SplitConfig


@dataclass
class SplitMetadata:
    """Metadata about a dataset split operation.

    Attributes:
        total_rows: Total number of rows in original dataset.
        train_rows: Number of rows in training split.
        validation_rows: Number of rows in validation split.
        test_rows: Number of rows in test split.
        date_range: Tuple of (min_date, max_date) for the dataset.
        train_ratios: Actual ratios achieved (may differ slightly from config).
        timestamp: When the split was performed.
    """

    total_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    date_range: tuple[datetime, datetime]
    train_date_boundary: datetime
    validation_date_boundary: datetime
    test_date_boundary: datetime
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format.

        Returns:
            Dictionary representation of metadata.
        """
        return {
            **asdict(self),
            "date_range": (
                self.date_range[0].isoformat(),
                self.date_range[1].isoformat(),
            ),
            "train_date_boundary": self.train_date_boundary.isoformat(),
            "validation_date_boundary": self.validation_date_boundary.isoformat(),
            "test_date_boundary": self.test_date_boundary.isoformat(),
            "timestamp": self.timestamp.isoformat(),
        }

    def describe(self) -> str:
        """Return a human-readable description of the split.

        Returns:
            Description string with split breakdown.
        """
        min_date, max_date = self.date_range
        total = self.total_rows
        return (
            f"Split Metadata: {total} rows from {min_date.date()} to {max_date.date()} | "
            f"Train: {self.train_rows} ({self.train_rows/total*100:.1f}%) | "
            f"Validation: {self.validation_rows} ({self.validation_rows/total*100:.1f}%) | "
            f"Test: {self.test_rows} ({self.test_rows/total*100:.1f}%)"
        )


@dataclass
class SplitResult:
    """Container for dataset split output.

    Attributes:
        train_df: Training dataset.
        validation_df: Validation dataset.
        test_df: Test dataset.
        metadata: Metadata about the split operation.
    """

    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    metadata: SplitMetadata

    def verify_no_overlap(self) -> bool:
        """Verify that train/validation/test sets have no overlapping indices.

        Returns:
            True if no overlap exists.

        Raises:
            ValueError: If overlaps are detected.
        """
        train_indices = set(self.train_df.index)
        val_indices = set(self.validation_df.index)
        test_indices = set(self.test_df.index)

        if train_indices & val_indices:
            raise ValueError("Overlap detected between train and validation sets")
        if train_indices & test_indices:
            raise ValueError("Overlap detected between train and test sets")
        if val_indices & test_indices:
            raise ValueError("Overlap detected between validation and test sets")

        return True

    def verify_complete_coverage(self, total_rows: int) -> bool:
        """Verify that splits cover all original rows exactly once.

        Args:
            total_rows: Total rows in original dataset.

        Returns:
            True if coverage is complete and non-overlapping.

        Raises:
            ValueError: If coverage is incomplete or duplicated.
        """
        split_rows = (
            len(self.train_df)
            + len(self.validation_df)
            + len(self.test_df)
        )
        if split_rows != total_rows:
            raise ValueError(
                f"Split coverage incomplete: {split_rows} split rows vs "
                f"{total_rows} original rows"
            )
        return True


class ChronologicalDatasetSplitter:
    """Splits datasets chronologically to prevent temporal data leakage.

    This splitter sorts matches by date and partitions them into training,
    validation, and test sets while preserving chronological order. This
    approach prevents the model from training on future data.

    Attributes:
        config: SplitConfig controlling the split ratios.
    """

    DATE_COLUMN = "Date"

    def __init__(self, config: Optional[SplitConfig] = None) -> None:
        """Initialize the chronological dataset splitter.

        Args:
            config: Split configuration. Defaults to standard 70/15/15 split.
        """
        self.config = config or SplitConfig()
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)

    def split(self, dataset: pd.DataFrame) -> SplitResult:
        """Split dataset chronologically into train/validation/test sets.

        Args:
            dataset: Input DataFrame containing match data with a Date column.

        Returns:
            SplitResult containing train_df, validation_df, test_df, and metadata.

        Raises:
            MissingDateColumnError: If Date column is missing.
            DatasetTooSmallError: If dataset has fewer than 3 rows.
            DuplicateMatchError: If duplicate match records are detected.
        """
        self.logger.info(
            f"Starting chronological split: {self.config.describe()}"
        )

        # Validate input
        self._validate_dataset(dataset)

        # Create working copy and sort by date
        working_df = dataset.copy()
        working_df[self.DATE_COLUMN] = pd.to_datetime(
            working_df[self.DATE_COLUMN]
        )
        working_df = working_df.sort_values(self.DATE_COLUMN).reset_index(
            drop=True
        )

        total_rows = len(working_df)
        self.logger.info(f"Dataset size: {total_rows} rows")

        # Calculate split boundaries
        train_size = int(total_rows * self.config.train_ratio)
        validation_size = int(
            total_rows * self.config.validation_ratio
        )
        # Remaining rows go to test to handle rounding
        test_size = total_rows - train_size - validation_size

        self.logger.info(
            f"Split sizes: train={train_size}, validation={validation_size}, "
            f"test={test_size}"
        )

        # Extract splits
        train_df = working_df.iloc[:train_size].copy()
        validation_df = working_df.iloc[
            train_size : train_size + validation_size
        ].copy()
        test_df = working_df.iloc[train_size + validation_size :].copy()

        # Build metadata
        date_range = (
            pd.to_datetime(working_df[self.DATE_COLUMN].min()),
            pd.to_datetime(working_df[self.DATE_COLUMN].max()),
        )
        train_date_boundary = (
            pd.to_datetime(train_df[self.DATE_COLUMN].max())
            if len(train_df) > 0
            else date_range[0]
        )
        validation_date_boundary = (
            pd.to_datetime(validation_df[self.DATE_COLUMN].max())
            if len(validation_df) > 0
            else train_date_boundary
        )
        test_date_boundary = (
            pd.to_datetime(test_df[self.DATE_COLUMN].max())
            if len(test_df) > 0
            else validation_date_boundary
        )

        metadata = SplitMetadata(
            total_rows=total_rows,
            train_rows=train_size,
            validation_rows=validation_size,
            test_rows=test_size,
            date_range=date_range,
            train_date_boundary=train_date_boundary,
            validation_date_boundary=validation_date_boundary,
            test_date_boundary=test_date_boundary,
        )

        result = SplitResult(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            metadata=metadata,
        )

        # Verify split integrity
        result.verify_no_overlap()
        result.verify_complete_coverage(total_rows)

        self.logger.info(metadata.describe())
        self.logger.info("Chronological split completed successfully")

        return result

    def _validate_dataset(self, dataset: pd.DataFrame) -> None:
        """Validate that dataset is suitable for splitting.

        Args:
            dataset: DataFrame to validate.

        Raises:
            MissingDateColumnError: If Date column is missing.
            DatasetTooSmallError: If dataset has fewer than 3 rows.
            DuplicateMatchError: If duplicate match records detected.
        """
        if dataset.empty:
            raise DatasetTooSmallError("Cannot split an empty dataset")

        if self.DATE_COLUMN not in dataset.columns:
            raise MissingDateColumnError(
                f"Dataset must contain '{self.DATE_COLUMN}' column for chronological sorting"
            )

        if len(dataset) < 3:
            raise DatasetTooSmallError(
                f"Dataset must have at least 3 rows for a meaningful split, "
                f"got {len(dataset)}"
            )

        # Check for duplicate match records
        self._check_duplicate_records(dataset)

    def _check_duplicate_records(self, dataset: pd.DataFrame) -> None:
        """Check for duplicate match records (same date, teams, and result).

        Raises:
            DuplicateMatchError: If duplicate records are found.
        """
        # Duplicate match detection: same date, homeTeam, awayTeam
        if "HomeTeam" in dataset.columns and "AwayTeam" in dataset.columns:
            potential_duplicates = dataset[
                [self.DATE_COLUMN, "HomeTeam", "AwayTeam"]
            ].duplicated(keep=False)

            if potential_duplicates.any():
                # Check if these are truly duplicates or different matches
                dup_rows = dataset[potential_duplicates].sort_values(
                    [self.DATE_COLUMN, "HomeTeam", "AwayTeam"]
                )

                # For same date/teams, check if other columns differ (different match)
                for idx in range(0, len(dup_rows) - 1, 2):
                    row1 = dup_rows.iloc[idx]
                    row2 = dup_rows.iloc[idx + 1]

                    if row1[self.DATE_COLUMN] == row2[self.DATE_COLUMN]:
                        # Same date and teams - check if it's truly a duplicate
                        if (
                            "FTHG" in dataset.columns
                            and "FTAG" in dataset.columns
                        ):
                            if (
                                row1.get("FTHG") == row2.get("FTHG")
                                and row1.get("FTAG") == row2.get("FTAG")
                            ):
                                raise DuplicateMatchError(
                                    f"Duplicate match record detected: "
                                    f"{row1['HomeTeam']} vs {row1['AwayTeam']} "
                                    f"on {row1[self.DATE_COLUMN]}"
                                )

    def describe(self) -> str:
        """Return a description of the splitter configuration.

        Returns:
            Description string.
        """
        return f"ChronologicalDatasetSplitter({self.config.describe()})"
