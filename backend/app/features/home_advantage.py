"""
Home and away venue advantage feature generator for MatchMind AI.

This module calculates venue-specific rolling performance features by using
TeamHistory as the source of historical match data.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional

import pandas as pd

from .base import FeatureGenerator
from .history import MatchHistory, TeamHistory
from .registry import FeatureRegistry


@FeatureRegistry.register
class HomeAdvantageGenerator(FeatureGenerator):
    """Generate venue-specific home and away performance features."""

    name: ClassVar[str] = "home_advantage"
    required_columns: ClassVar[List[str]] = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    ]
    output_columns: ClassVar[List[str]] = [
        "home_home_win_rate",
        "away_away_win_rate",
        "home_home_draw_rate",
        "away_away_draw_rate",
        "home_home_loss_rate",
        "away_away_loss_rate",
        "home_home_goals_per_match",
        "away_away_goals_per_match",
        "home_home_goals_against_per_match",
        "away_away_goals_against_per_match",
        "home_home_points_per_match",
        "away_away_points_per_match",
        "home_home_clean_sheet_rate",
        "away_away_clean_sheet_rate",
        "home_home_failed_to_score_rate",
        "away_away_failed_to_score_rate",
    ]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        team_history: Optional[TeamHistory] = None,
    ) -> None:
        """Initialize the home advantage generator.

        Args:
            config: Optional configuration. Supports ``window``.
            team_history: Optional injected TeamHistory dependency.
        """
        super().__init__(config=config)
        self.window = self._get_window()
        self.team_history = team_history

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate venue-specific team performance features.

        Args:
            df: Input match dataset.

        Returns:
            DataFrame containing generated home advantage feature columns.

        Raises:
            ValueError: If required input columns are missing or ``window`` is
                less than one.
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
                venue="home",
            )
            away_stats = team_history.get_team_statistics(
                team=str(row["AwayTeam"]),
                before_date=row["Date"],
                limit=self.window,
                venue="away",
            )

            self._assign_team_features(features_df, index, "home_home", home_stats)
            self._assign_team_features(features_df, index, "away_away", away_stats)

        self.logger.info(f"Successfully generated {self.name} features")
        return features_df

    def _get_window(self) -> int:
        """Return the configured rolling window size."""
        window = int(self.config.get("window", 5))
        if window < 1:
            raise ValueError("Home advantage rolling window must be at least 1")
        return window

    def _assign_team_features(
        self,
        features_df: pd.DataFrame,
        index: Any,
        prefix: str,
        stats: Dict[str, float],
    ) -> None:
        """Assign venue-specific TeamHistory statistics to feature columns.

        Args:
            features_df: Output feature DataFrame being populated.
            index: Source row index to assign.
            prefix: Feature prefix, either ``home_home`` or ``away_away``.
            stats: Team statistics returned by TeamHistory.
        """
        features_df.loc[index, f"{prefix}_win_rate"] = stats["win_rate"]
        features_df.loc[index, f"{prefix}_draw_rate"] = stats["draw_rate"]
        features_df.loc[index, f"{prefix}_loss_rate"] = stats["loss_rate"]
        features_df.loc[index, f"{prefix}_goals_per_match"] = stats[
            "goals_per_match"
        ]
        features_df.loc[index, f"{prefix}_goals_against_per_match"] = stats[
            "goals_against_per_match"
        ]
        features_df.loc[index, f"{prefix}_points_per_match"] = stats[
            "points_per_match"
        ]
        features_df.loc[index, f"{prefix}_clean_sheet_rate"] = stats[
            "clean_sheet_rate"
        ]
        features_df.loc[index, f"{prefix}_failed_to_score_rate"] = stats[
            "failed_to_score_rate"
        ]
