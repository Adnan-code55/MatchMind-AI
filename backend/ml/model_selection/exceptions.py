"""
Custom exceptions for the model selection module.
"""

class ModelSelectionError(Exception):
    """Base exception for all model selection-related errors."""
    pass


class NoModelsTrainedError(ModelSelectionError):
    """Raised when the training step yields no valid models."""
    pass


class InvalidModelConfigError(ModelSelectionError):
    """Raised when the provided models list or configuration is invalid."""
    pass
