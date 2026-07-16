"""
Data Splitting Engine module.

Provides dataset splitting capabilities for preparing machine learning datasets.
"""

from .exceptions import (
    DatasetTooSmallError,
    InvalidSplitRatioError,
    SplittingError,
    StratificationError,
)
from .metadata import SplitMetadata
from .splitter import DatasetSplitter

__all__ = [
    "DatasetSplitter",
    "SplitMetadata",
    "SplittingError",
    "InvalidSplitRatioError",
    "DatasetTooSmallError",
    "StratificationError",
]
