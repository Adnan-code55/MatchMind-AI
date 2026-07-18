"""
Metadata definitions for the Prediction module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any


@dataclass
class PredictionMetadata:
    """Metadata recorded for a prediction."""
    model_name: str
    inference_time_ms: float
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "model_name": self.model_name,
            "inference_time_ms": self.inference_time_ms,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
        }
