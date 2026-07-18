"""
MatchMind AI Evaluation Module.

Provides tools for evaluating machine learning models, validating predictions,
calculating metrics, exporting reports, and comparing models.
"""

from .exceptions import (
    EvaluationError,
    InvalidInputError,
    MetricComputationError,
    MissingMetricError
)
from .metadata import ModelMetadata, EvaluationMetadata
from .evaluator import EvaluationEngine
from .comparison import ModelComparator
from .reports import export_to_json, export_to_csv
from .metrics import compute_classification_metrics

__all__ = [
    "EvaluationEngine",
    "ModelComparator",
    "ModelMetadata",
    "EvaluationMetadata",
    "export_to_json",
    "export_to_csv",
    "compute_classification_metrics",
    "EvaluationError",
    "InvalidInputError",
    "MetricComputationError",
    "MissingMetricError"
]
