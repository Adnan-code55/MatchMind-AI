"""
Custom exceptions for the Data Splitting Engine.

This module defines domain-specific exceptions for dataset splitting,
validation, and metadata generation.
"""


class SplittingError(Exception):
    """Base exception for splitting errors."""


class InvalidSplitRatioError(SplittingError):
    """Raised when split ratios are invalid or do not sum to 1.0."""


class DatasetTooSmallError(SplittingError):
    """Raised when the dataset is too small for the requested split."""


class StratificationError(SplittingError):
    """Raised when stratification cannot be performed."""
