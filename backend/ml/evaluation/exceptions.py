"""
Custom exceptions for the evaluation module.
"""

class EvaluationError(Exception):
    """Base exception for all evaluation-related errors."""
    pass


class InvalidInputError(EvaluationError):
    """Raised when input shapes, types, or values are invalid for evaluation."""
    pass


class MetricComputationError(EvaluationError):
    """Raised when a specific metric fails to compute."""
    pass

class MissingMetricError(EvaluationError):
    """Raised when an expected metric is missing during comparison or reporting."""
    pass
