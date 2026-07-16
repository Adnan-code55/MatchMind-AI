"""
Metadata tracking for the Feature Engineering Engine.

This module defines data structures for tracking feature engineering operations
and maintaining traceability in machine learning pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class FeatureMetadata:
    """Metadata about a feature engineering operation.

    Attributes:
        window_size: The rolling window size used for generating historical features.
        features_generated: List of column names that were added to the dataset.
        initial_feature_count: Number of features in the dataset before processing.
        final_feature_count: Number of features in the dataset after processing.
        timestamp: When the feature engineering was performed.
    """

    window_size: int
    features_generated: List[str]
    initial_feature_count: int
    final_feature_count: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format.

        Returns:
            Dictionary representation of metadata.
        """
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def describe(self) -> str:
        """Return human-readable description of the feature engineering operation.

        Returns:
            Description string.
        """
        desc = (
            f"FeatureMetadata: Window={self.window_size} | "
            f"Features Added={len(self.features_generated)} | "
            f"Total Features={self.final_feature_count}"
        )
        return desc
