"""
Metadata classes for the evaluation module.

Captures context about the evaluation run.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List


@dataclass
class ModelMetadata:
    """Metadata about the model being evaluated."""
    name: str
    version: str
    hyperparameters: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationMetadata:
    """Metadata about an evaluation run.

    Attributes:
        model: Metadata about the model.
        dataset_name: Name or identifier of the dataset used.
        num_samples: Total number of samples evaluated.
        split_strategy: e.g., 'KFold', 'StratifiedKFold', 'TrainTestSplit'.
        timestamp: When the evaluation was performed.
        extra: Any additional metadata.
    """
    model: ModelMetadata
    dataset_name: str
    num_samples: int
    split_strategy: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: Dict[str, Any] = field(default_factory=dict)
