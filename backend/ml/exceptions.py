"""
Custom exceptions for the ML module.

This module defines domain-specific exceptions for dataset splitting,
model training, and related machine learning operations.
"""


class DatasetSplitError(Exception):
    """Base exception for dataset splitting errors."""


class InvalidSplitConfiguration(DatasetSplitError):
    """Raised when split ratios are invalid."""


class DatasetTooSmallError(DatasetSplitError):
    """Raised when dataset has insufficient rows to perform split."""


class MissingDateColumnError(DatasetSplitError):
    """Raised when required date column is missing from dataset."""


class DuplicateMatchError(DatasetSplitError):
    """Raised when duplicate match records are detected."""
