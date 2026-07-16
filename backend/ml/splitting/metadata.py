"""
Metadata tracking for the Data Splitting Engine.

This module defines data structures for tracking splitting operations
and maintaining traceability in machine learning pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class SplitMetadata:
    """Metadata about a dataset split operation.

    Attributes:
        train_count: Number of samples in training set.
        test_count: Number of samples in test set.
        val_count: Number of samples in validation set (if applicable).
        split_ratios: Ratios used for the split (e.g., {'test': 0.2, 'val': 0.1}).
        random_seed: Random seed used for shuffling/splitting.
        feature_count: Number of features in the dataset.
        target_column: Name of target column used for stratification (if applicable).
        timestamp: When the split was performed.
    """

    train_count: int
    test_count: int
    val_count: int
    split_ratios: Dict[str, float]
    random_seed: int
    feature_count: int
    target_column: Optional[str] = None
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
        """Return human-readable description of the split.

        Returns:
            Description string.
        """
        desc = (
            f"SplitMetadata: Train={self.train_count} | "
            f"Test={self.test_count}"
        )
        if self.val_count > 0:
            desc += f" | Val={self.val_count}"
        
        desc += f" | Features={self.feature_count}"
        
        if self.target_column:
            desc += f" | Stratified on='{self.target_column}'"
            
        return desc
