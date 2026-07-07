"""Custom exceptions for Dataset Builder operations."""

from __future__ import annotations


class DatasetBuilderError(Exception):
    """Base exception for dataset building failures."""


class MissingFeatureError(DatasetBuilderError):
    """Raised when required fields are missing from input records."""


class InvalidMatchError(DatasetBuilderError):
    """Raised when a match record cannot be validated or target labels cannot be derived."""


class DuplicateMatchError(DatasetBuilderError):
    """Raised when duplicate match records are detected in the input."""


class EmptyDatasetError(DatasetBuilderError):
    """Raised when the provided input dataset is empty."""
