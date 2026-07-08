"""
Configuration for chronological dataset splitting.

This module defines the SplitConfig dataclass that validates and stores
parameters for chronological dataset partitioning.
"""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import InvalidSplitConfiguration


@dataclass
class SplitConfig:
    """Configuration for chronological dataset splitting.

    Attributes:
        train_ratio: Fraction of data for training (default 0.70).
        validation_ratio: Fraction of data for validation (default 0.15).
        test_ratio: Fraction of data for testing (default 0.15).
        shuffle: Whether to shuffle data within splits (default False for
            chronological order preservation).

    Raises:
        InvalidSplitConfiguration: If ratios don't sum to 1.0 or are non-positive.
    """

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle: bool = False

    def __post_init__(self) -> None:
        """Validate split configuration after initialization."""
        self._validate_ratios()

    def _validate_ratios(self) -> None:
        """Validate that ratios sum to 1.0 and are all positive.

        Raises:
            InvalidSplitConfiguration: If ratios are invalid.
        """
        total = self.train_ratio + self.validation_ratio + self.test_ratio

        # Check sum with small tolerance for floating-point precision
        if not (0.9999 < total < 1.0001):
            raise InvalidSplitConfiguration(
                f"Split ratios must sum to 1.0, got {total:.4f}"
            )

        if self.train_ratio <= 0:
            raise InvalidSplitConfiguration(
                f"Train ratio must be positive, got {self.train_ratio}"
            )

        if self.validation_ratio < 0:
            raise InvalidSplitConfiguration(
                f"Validation ratio must be non-negative, got {self.validation_ratio}"
            )

        if self.test_ratio < 0:
            raise InvalidSplitConfiguration(
                f"Test ratio must be non-negative, got {self.test_ratio}"
            )

    def describe(self) -> str:
        """Return a human-readable description of the split configuration.

        Returns:
            Description string with all ratios and shuffle setting.
        """
        return (
            f"SplitConfig(train={self.train_ratio:.2%}, "
            f"validation={self.validation_ratio:.2%}, "
            f"test={self.test_ratio:.2%}, shuffle={self.shuffle})"
        )
