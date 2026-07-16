"""
Feature Engineering module for MatchMind AI.

This module provides tools for generating historical and rolling statistics
from match data to be used as machine learning features.
"""

from .exceptions import (
    FeatureEngineeringError,
    InvalidWindowError,
    MissingColumnError,
)
from .metadata import FeatureMetadata
from .statistics_engine import StatisticsEngine
from .validators import validate_required_columns

__all__ = [
    "FeatureEngineeringError",
    "InvalidWindowError",
    "MissingColumnError",
    "FeatureMetadata",
    "StatisticsEngine",
    "validate_required_columns",
]
