"""
Exceptions for the Prediction module.
"""

class PredictionError(Exception):
    """Base exception for all prediction-related errors."""
    pass


class ModelLoadError(PredictionError):
    """Raised when the model cannot be loaded from disk."""
    pass


class InvalidInputDataError(PredictionError):
    """Raised when the input data provided for prediction is invalid."""
    pass


class FeatureMismatchError(PredictionError):
    """Raised when the input data does not match the expected features."""
    pass
