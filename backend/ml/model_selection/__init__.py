"""
MatchMind AI Model Selection Module.

Provides tools for automating model training, evaluation, ranking, selection,
and persistence.
"""

from .exceptions import ModelSelectionError, NoModelsTrainedError, InvalidModelConfigError
from .metadata import ModelSelectionMetadata
from .validators import validate_models_list, validate_selection_dataset
from .ranking import ModelRanker
from .persistence import persist_best_model
from .selector import ModelSelector

__all__ = [
    "ModelSelector",
    "ModelSelectionMetadata",
    "ModelRanker",
    "persist_best_model",
    "validate_models_list",
    "validate_selection_dataset",
    "ModelSelectionError",
    "NoModelsTrainedError",
    "InvalidModelConfigError"
]
