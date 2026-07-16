"""
Custom exceptions for the Feature Engineering Engine.

This module defines domain-specific exceptions for feature generation,
validation, and metadata tracking.
"""


class FeatureEngineeringError(Exception):
    """Base exception for feature engineering errors."""


class MissingColumnError(FeatureEngineeringError):
    """Raised when required columns are missing from the dataset."""


class InvalidWindowError(FeatureEngineeringError):
    """Raised when the rolling window size is invalid (e.g., less than 1)."""
