"""
Public API for the ML Training module.
"""

from .trainer import ModelTrainer
from .metadata import TrainingMetadata
from .exceptions import (
    TrainingError,
    UnsupportedModelError,
    EmptyDatasetError,
    MismatchedDimensionsError,
    InvalidInputTypeError,
)

__all__ = [
    "ModelTrainer",
    "TrainingMetadata",
    "TrainingError",
    "UnsupportedModelError",
    "EmptyDatasetError",
    "MismatchedDimensionsError",
    "InvalidInputTypeError",
]
