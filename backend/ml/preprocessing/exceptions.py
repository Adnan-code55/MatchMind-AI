"""
Custom exceptions for the preprocessing module.

This module defines domain-specific exceptions for data preprocessing,
feature transformation, and related operations.
"""


class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""


class MissingTargetError(PreprocessingError):
    """Raised when target column is missing or invalid."""


class InvalidFeatureTypeError(PreprocessingError):
    """Raised when feature type is not recognized or supported."""


class UnsupportedTransformationError(PreprocessingError):
    """Raised when requested transformation is not supported."""


class FeatureDetectionError(PreprocessingError):
    """Raised when feature detection fails."""


class TransformationStateError(PreprocessingError):
    """Raised when transformer is used before fitting."""
