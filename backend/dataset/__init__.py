"""Dataset builder package for MatchMind AI.

This package provides a builder for converting processed match records into a
machine-learning-ready dataset.
"""

from .dataset_builder import DatasetBuilder
from .exceptions import (
    DatasetBuilderError,
    DuplicateMatchError,
    EmptyDatasetError,
    InvalidMatchError,
    MissingFeatureError,
)

__all__ = [
    "DatasetBuilder",
    "DatasetBuilderError",
    "DuplicateMatchError",
    "EmptyDatasetError",
    "InvalidMatchError",
    "MissingFeatureError",
]
