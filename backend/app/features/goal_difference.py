"""
Goal difference feature generator for MatchMind AI.

This module calculates rolling goal difference features for home and away teams
using the History Engine as the single source of historical match data.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional

import pandas as pd

from .base import FeatureGenerator
from .history import MatchHistory, TeamHistory
from .registry import FeatureRegistry


@FeatureRegistry.register
class GoalDifferenceGenerator(FeatureGenerator):
    """Generate rolling goal difference features from team history."""

    name: ClassVar[str] = "goal_difference"
    required_columns: ClassVar[List[str]] = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    ]
    output_columns: ClassVar[List[str]] = [
        "home_goals_for",
        "home_goals_against",
        "home_goal_difference",
        "home_average_goal_difference",
        "away_goals_for",
        "away_goals_against",
        "away_goal_difference",
        "away_average_goal_difference",
    ]
    dependencies: ClassVar[List[str]] = []

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        team_history: Optional[TeamHistory] = None,
    ) -> None:
        """Initialize the goal difference generator.

        Args:
            config: Optional generator configuration. Supports ``window``.
            team_history: Optional injected TeamHistory dependency for tests or
                pre-built history services.
        """
        super().__init__(config=config)
        self.window = self._get_window()
        self.team_history = team_history

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling goal difference features.

        Args:
            df: Input match dataset.

        Returns:
            DataFrame containing only generated goal difference feature columns.

        Raises:
            ValueError: If required input columns are missing or the configured
                window is invalid.
        """
        self.validate_input(df)
        self.logger.info(
            f"Generating {self.name} features for {len(df)} matches with window={self.window}"
        )

        working_df = df.copy()
        working_df["Date"] = pd.to_datetime(working_df["Date"])
        team_history = self.team_history or TeamHistory(MatchHistory(working_df))

        features_df = pd.DataFrame(index=working_df.index)
        features_df[self.output_columns] = 0.0

        for index, row in working_df.iterrows():
            home_stats = team_history.get_team_statistics(
                team=str(row["HomeTeam"]),
                before_date=row["Date"],
                limit=self.window,
            )
            away_stats = team_history.get_team_statistics(
                team=str(row["AwayTeam"]),
                before_date=row["Date"],
                limit=self.window,
            )

            self._assign_team_features(features_df, index, "home", home_stats)
            self._assign_team_features(features_df, index, "away", away_stats)

        self.logger.info(f"Successfully generated {self.name} features")
        return features_df

    def _get_window(self) -> int:
        """Return the configured rolling window size."""
        window = int(self.config.get("window", 5))
        if window < 1:
            raise ValueError("Goal difference rolling window must be at least 1")
        return window

    def _assign_team_features(
        self,
        features_df: pd.DataFrame,
        index: Any,
        prefix: str,
        stats: Dict[str, float],
    ) -> None:
        """Assign team goal statistics into feature columns.

        Args:
            features_df: Output feature DataFrame being populated.
            index: Source row index to assign.
            prefix: ``home`` or ``away`` feature prefix.
            stats: Team statistics returned by TeamHistory.
        """
        matches_played = stats["matches_played"]
        goal_difference = stats["goal_difference"]
        average_goal_difference = (
            goal_difference / matches_played if matches_played else 0.0
        )

        features_df.loc[index, f"{prefix}_goals_for"] = stats["goals_for"]
        features_df.loc[index, f"{prefix}_goals_against"] = stats["goals_against"]
        features_df.loc[index, f"{prefix}_goal_difference"] = goal_difference
        features_df.loc[
            index,
            f"{prefix}_average_goal_difference",
        ] = average_goal_difference
