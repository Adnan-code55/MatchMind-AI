"""
Metadata tracking for preprocessing pipeline.

This module defines data structures for tracking preprocessing state,
transformations applied, and pipeline configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class FeatureInfo:
    """Information about a feature's type and handling.

    Attributes:
        name: Feature column name.
        feature_type: Type classification (numerical, categorical, date, metadata).
        original_dtype: Original pandas dtype.
        missing_count: Number of missing values.
        missing_strategy: Strategy used for handling missing values.
        encoded: Whether feature was encoded.
        scaled: Whether feature was scaled.
    """

    name: str
    feature_type: str
    original_dtype: str
    missing_count: int
    missing_strategy: Optional[str] = None
    encoded: bool = False
    scaled: bool = False
    encoding_type: Optional[str] = None
    categories: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass
class PreprocessingMetadata:
    """Metadata about preprocessing operations.

    Attributes:
        original_shape: Original dataset shape (rows, columns).
        processed_shape: Processed dataset shape.
        target_column: Name of target column.
        target_dtype: Original dtype of target column.
        features: List of FeatureInfo for each feature.
        encoded_features: Names of encoded features.
        scaled_features: Names of scaled features.
        dropped_features: Names of dropped features.
        timestamp: When preprocessing was performed.
        pipeline_config: Configuration used for pipeline.
    """

    original_shape: tuple[int, int]
    processed_shape: tuple[int, int]
    target_column: Optional[str]
    target_dtype: Optional[str]
    features: List[FeatureInfo] = field(default_factory=list)
    encoded_features: List[str] = field(default_factory=list)
    scaled_features: List[str] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    pipeline_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format.

        Returns:
            Dictionary representation of metadata.
        """
        return {
            "original_shape": self.original_shape,
            "processed_shape": self.processed_shape,
            "target_column": self.target_column,
            "target_dtype": self.target_dtype,
            "features": [f.to_dict() for f in self.features],
            "encoded_features": self.encoded_features,
            "scaled_features": self.scaled_features,
            "dropped_features": self.dropped_features,
            "timestamp": self.timestamp.isoformat(),
            "pipeline_config": self.pipeline_config,
        }

    def describe(self) -> str:
        """Return human-readable description of preprocessing.

        Returns:
            Description string.
        """
        return (
            f"PreprocessingMetadata: {self.original_shape[0]} rows × {self.original_shape[1]} cols "
            f"→ {self.processed_shape[0]} rows × {self.processed_shape[1]} cols | "
            f"Target: {self.target_column} | "
            f"Encoded: {len(self.encoded_features)} | "
            f"Scaled: {len(self.scaled_features)} | "
            f"Dropped: {len(self.dropped_features)}"
        )
