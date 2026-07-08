"""
MatchMind AI machine learning module.

This package provides ML utilities including chronological dataset splitting,
model training support, and evaluation tools designed for football prediction.
"""

from .dataset_splitter import (
    ChronologicalDatasetSplitter,
    SplitMetadata,
    SplitResult,
)
from .exceptions import (
    DatasetSplitError,
    DatasetTooSmallError,
    DuplicateMatchError,
    InvalidSplitConfiguration,
    MissingDateColumnError,
)
from .split_config import SplitConfig

__all__ = [
    "ChronologicalDatasetSplitter",
    "SplitConfig",
    "SplitResult",
    "SplitMetadata",
    "DatasetSplitError",
    "InvalidSplitConfiguration",
    "DatasetTooSmallError",
    "MissingDateColumnError",
    "DuplicateMatchError",
]
