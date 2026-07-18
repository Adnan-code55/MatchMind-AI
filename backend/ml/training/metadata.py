"""
Metadata definitions for the ML Training module.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class TrainingMetadata:
    """Metadata recorded for a trained model."""
    algorithm: str
    dataset_size: int
    feature_count: int
    training_score: float
    validation_score: float
    parameters: Dict[str, Any]
    random_seed: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "algorithm": self.algorithm,
            "dataset_size": self.dataset_size,
            "feature_count": self.feature_count,
            "training_score": self.training_score,
            "validation_score": self.validation_score,
            "parameters": self.parameters,
            "random_seed": self.random_seed,
            "timestamp": self.timestamp.isoformat(),
        }
