"""
MatchMind AI feature engineering package.

This package exposes the architecture for feature generation, registration,
pipeline execution, and feature combination.
"""

from .base import FeatureGenerator
from .combiner import FeatureCombiner
from .pipeline import FeaturePipeline
from .registry import FeatureRegistry

__all__ = [
    "FeatureGenerator",
    "FeatureCombiner",
    "FeaturePipeline",
    "FeatureRegistry",
]
