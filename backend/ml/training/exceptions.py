"""
Custom exceptions for the ML Training module.
"""

class TrainingError(Exception):
    """Base exception for all training-related errors."""

class UnsupportedModelError(TrainingError):
    """Raised when an unsupported model name is provided."""

class EmptyDatasetError(TrainingError):
    """Raised when a provided dataset has no samples or features."""

class MismatchedDimensionsError(TrainingError):
    """Raised when features and targets have different number of samples."""

class InvalidInputTypeError(TrainingError):
    """Raised when inputs are not pandas DataFrames/Series or other accepted types."""
