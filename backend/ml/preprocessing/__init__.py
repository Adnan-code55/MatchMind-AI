"""
Machine learning preprocessing module.

This package provides a comprehensive, reusable preprocessing pipeline
for preparing football data for machine learning models.
"""

from .exceptions import (
    FeatureDetectionError,
    InvalidFeatureTypeError,
    MissingTargetError,
    PreprocessingError,
    TransformationStateError,
    UnsupportedTransformationError,
)
from .metadata import FeatureInfo, PreprocessingMetadata
from .preprocessing_pipeline import FeatureDetector, PreprocessingPipeline
from .transformers import (
    CategoricalEncoder,
    FeatureScaler,
    MissingValueHandler,
    TargetExtractor,
    Transformer,
)

__all__ = [
    "PreprocessingPipeline",
    "FeatureDetector",
    "Transformer",
    "MissingValueHandler",
    "CategoricalEncoder",
    "FeatureScaler",
    "TargetExtractor",
    "PreprocessingMetadata",
    "FeatureInfo",
    "PreprocessingError",
    "MissingTargetError",
    "InvalidFeatureTypeError",
    "UnsupportedTransformationError",
    "FeatureDetectionError",
    "TransformationStateError",
]
