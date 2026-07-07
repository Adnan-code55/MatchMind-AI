"""
Feature engineering pipeline orchestrator for MatchMind AI.

This module defines the feature pipeline that coordinates generator execution,
ordering, and safe combination of derived features with the base dataset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..config.settings import PipelineSettings
from ..data.logger import PipelineLogger
from .base import FeatureGenerator
from .combiner import FeatureCombiner
from .registry import FeatureRegistry


class FeaturePipeline:
    """Pipeline that executes registered feature generators."""

    def __init__(
        self,
        registry: FeatureRegistry,
        combiner: Optional[FeatureCombiner] = None,
        settings: Optional[PipelineSettings] = None,
        feature_order: Optional[List[str]] = None,
        generator_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Initialize the feature engineering pipeline."""
        self.registry = registry
        self.settings = settings
        self.feature_order = feature_order
        self.generator_configs = generator_configs or {}
        self.combiner = combiner or FeatureCombiner()
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)
        self.generators = self._build_generators()

    def _build_generators(self) -> List[FeatureGenerator]:
        """Instantiate feature generators in the configured order."""
        self.logger.info("Building feature generators from registry.")
        self.registry.discover_generators()
        instances = self.registry.instantiate_all(
            configs=self.generator_configs,
            order=self.feature_order,
        )
        return instances

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the feature pipeline and return the augmented dataset."""
        current_df = df.copy()

        for generator in self.generators:
            if not generator.supports(current_df):
                self.logger.warning(
                    f"Skipping generator '{generator.name}' because required columns are missing."
                )
                continue

            self.logger.info(f"Executing feature generator '{generator.name}'.")
            generator.validate_input(current_df)
            feature_df = generator.generate(current_df)
            feature_df = self._validate_generator_output(generator, feature_df)
            current_df = self.combiner.combine(current_df, feature_df)

        return current_df

    def _validate_generator_output(
        self,
        generator: FeatureGenerator,
        feature_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate generated feature output before combination."""
        if feature_df.empty:
            self.logger.warning(
                f"Generator '{generator.name}' produced no features."
            )
        return feature_df

    def describe(self) -> str:
        """Return a summary of the configured feature pipeline."""
        return (
            f"FeaturePipeline(order={self.feature_order}, "
            f"generators={[generator.name for generator in self.generators]}, "
            f"combiner={self.combiner.strategy})"
        )
