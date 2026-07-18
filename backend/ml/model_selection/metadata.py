"""
Metadata classes for the model selection module.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List


@dataclass
class ModelSelectionMetadata:
    """Metadata about the model selection run.

    Attributes:
        winning_model: The name of the best performing model.
        evaluation_metrics: The metrics of the winning model.
        ranking: A full ranked list of all evaluated models and their metrics.
        timestamp: When the selection was performed.
        extra: Any additional context.
    """
    winning_model: str
    evaluation_metrics: Dict[str, Any]
    ranking: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: Dict[str, Any] = field(default_factory=dict)
