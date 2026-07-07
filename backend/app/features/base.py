"""
Base abstractions for MatchMind AI feature engineering.

This module defines the contract and shared utilities for feature generator
implementations that will produce derived feature sets from match datasets.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Dict, List, Optional

import pandas as pd

from ..data.logger import PipelineLogger


class FeatureGenerator(abc.ABC):
    """Abstract base class for feature generation components."""

    name: ClassVar[str] = "base_feature"
    required_columns: ClassVar[List[str]] = []
    output_columns: ClassVar[List[str]] = []
    dependencies: ClassVar[List[str]] = []

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize a feature generator with optional runtime configuration."""
        self.config = config or {}
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)

    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> None:
        """Validate that required source columns exist in the dataset."""
        missing_columns = [column for column in cls.required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Feature generator '{cls.name}' is missing required columns: {missing_columns}"
            )

    def supports(self, df: pd.DataFrame) -> bool:
        """Return whether this generator can run against the provided dataset."""
        return all(column in df.columns for column in self.required_columns)

    def validate_input(self, df: pd.DataFrame) -> None:
        """Validate runtime input before generation begins."""
        self.validate_schema(df)

    @abc.abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate derived features from the input dataset."""
        raise NotImplementedError("Feature generators must implement generate().")

    def describe(self) -> str:
        """Return a human-readable description of the generator."""
        return (
            f"FeatureGenerator(name={self.name}, "
            f"required_columns={self.required_columns}, "
            f"output_columns={self.output_columns}, "
            f"dependencies={self.dependencies})"
        )
