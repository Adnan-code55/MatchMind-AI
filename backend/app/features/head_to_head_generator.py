"""
Head-to-head feature generator for MatchMind AI.

This module maps aggregate head-to-head history into model-ready feature
columns without directly scanning historical match data.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional

import pandas as pd

from .base import FeatureGenerator
from .history import HeadToHeadHistory, MatchHistory
from .registry import FeatureRegistry


@FeatureRegistry.register
class HeadToHeadGenerator(FeatureGenerator):
    """Generate rolling head-to-head features for each fixture."""

    name: ClassVar[str] = "head_to_head"
    required_columns: ClassVar[List[str]] = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
    ]
    output_columns: ClassVar[List[str]] = [
        "home_h2h_wins",
        "away_h2h_wins",
        "h2h_draws",
        "home_h2h_win_rate",
        "away_h2h_win_rate",
        "h2h_draw_rate",
        "home_h2h_goals",
        "away_h2h_goals",
        "h2h_average_goals",
        "recent_h2h_goal_difference",
        "h2h_matches_played",
    ]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        head_to_head_history: Optional[HeadToHeadHistory] = None,
    ) -> None:
        """Initialize the head-to-head generator.

        Args:
            config: Optional configuration. Supports ``window``.
            head_to_head_history: Optional injected HeadToHeadHistory
                dependency.
        """
        super().__init__(config=config)
        self.window = self._get_window()
        self.head_to_head_history = head_to_head_history

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate head-to-head features for each match.

        Args:
            df: Input match dataset.

        Returns:
            DataFrame containing generated head-to-head feature columns.

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
        h2h_history = self.head_to_head_history or HeadToHeadHistory(
            MatchHistory(working_df)
        )

        features_df = pd.DataFrame(index=working_df.index)
        features_df[self.output_columns] = 0.0

        for index, row in working_df.iterrows():
            stats = h2h_history.get_head_to_head_statistics(
                home_team=str(row["HomeTeam"]),
                away_team=str(row["AwayTeam"]),
                before_date=row["Date"],
                limit=self.window,
            )
            self._assign_features(features_df, index, stats)

        self.logger.info(f"Successfully generated {self.name} features")
        return features_df

    def _get_window(self) -> int:
        """Return the configured rolling window size."""
        window = int(self.config.get("window", 5))
        if window < 1:
            raise ValueError("Head-to-head rolling window must be at least 1")
        return window

    def _assign_features(
        self,
        features_df: pd.DataFrame,
        index: Any,
        stats: Dict[str, float],
    ) -> None:
        """Assign head-to-head statistics to feature columns.

        Args:
            features_df: Output feature DataFrame being populated.
            index: Source row index to assign.
            stats: Statistics returned by HeadToHeadHistory.
        """
        matches = stats["matches"]
        features_df.loc[index, "home_h2h_wins"] = stats["home_wins"]
        features_df.loc[index, "away_h2h_wins"] = stats["away_wins"]
        features_df.loc[index, "h2h_draws"] = stats["draws"]
        features_df.loc[index, "home_h2h_win_rate"] = (
            stats["home_wins"] / matches if matches else 0.0
        )
        features_df.loc[index, "away_h2h_win_rate"] = (
            stats["away_wins"] / matches if matches else 0.0
        )
        features_df.loc[index, "h2h_draw_rate"] = (
            stats["draws"] / matches if matches else 0.0
        )
        features_df.loc[index, "home_h2h_goals"] = stats["home_goals"]
        features_df.loc[index, "away_h2h_goals"] = stats["away_goals"]
        features_df.loc[index, "h2h_average_goals"] = stats["average_goals"]
        features_df.loc[index, "recent_h2h_goal_difference"] = stats[
            "goal_difference"
        ]
        features_df.loc[index, "h2h_matches_played"] = matches
