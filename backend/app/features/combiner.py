"""
Feature combination strategies for MatchMind AI.

This module provides the extensible strategy for combining generated feature
frames with the source dataset in a safe and reproducible way.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


CombineStrategy = Literal["concat", "merge"]


class FeatureCombiner:
    """Combine generated feature outputs with the base dataset."""

    def __init__(self, strategy: CombineStrategy = "concat") -> None:
        """Initialize the combiner with a chosen strategy."""
        self.strategy = strategy

    def combine(self, base_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Combine the base dataset with derived feature output."""
        if feature_df.empty:
            return base_df.copy()

        if self.strategy == "concat":
            return pd.concat([base_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

        if self.strategy == "merge":
            return base_df.merge(feature_df, left_index=True, right_index=True, how="left")

        raise ValueError(
            f"Unsupported feature combiner strategy: {self.strategy}"
        )

    def validate(self, base_df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
        """Validate that combining the feature output will not conflict with base columns."""
        conflicting_columns = set(base_df.columns).intersection(feature_df.columns)
        if conflicting_columns:
            raise ValueError(
                f"Feature output contains columns that conflict with existing dataset: {sorted(conflicting_columns)}"
            )
