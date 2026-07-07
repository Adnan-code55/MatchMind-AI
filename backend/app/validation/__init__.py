"""
Validation package for MatchMind AI.

This package provides dataset validation, feature statistics, and validation
report generation for engineered datasets prior to model training.
"""

from .dataset_report import DatasetValidationReport
from .feature_validator import FeatureValidator
from .statistics import ValidationStatistics

__all__ = [
    "DatasetValidationReport",
    "FeatureValidator",
    "ValidationStatistics",
]
