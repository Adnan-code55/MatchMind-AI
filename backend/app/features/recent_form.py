"""
Recent form feature generator for MatchMind AI.

This module computes rolling form statistics for teams based on their last 5
matches. Features include points, wins, draws, losses, goals scored, and goals
conceded. The implementation works chronologically to ensure no future information
is used.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional

import pandas as pd
import numpy as np

from ..data.logger import PipelineLogger
from .base import FeatureGenerator
from .registry import FeatureRegistry


MODULE_NAME = "RecentFormGenerator"


@FeatureRegistry.register
class RecentFormGenerator(FeatureGenerator):
    """
    Generate rolling form statistics based on the last 5 matches.

    This generator calculates team form features by examining each team's
    performance in their previous 5 matches. All calculations work chronologically
    to ensure that only past information is used for each match.
    """

    name: ClassVar[str] = "recent_form"
    required_columns: ClassVar[List[str]] = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    ]
    output_columns: ClassVar[List[str]] = [
        "home_form_points_last5",
        "away_form_points_last5",
        "home_wins_last5",
        "away_wins_last5",
        "home_draws_last5",
        "away_draws_last5",
        "home_losses_last5",
        "away_losses_last5",
        "home_goals_scored_last5",
        "away_goals_scored_last5",
        "home_goals_conceded_last5",
        "away_goals_conceded_last5",
        "home_goal_difference_last5",
        "away_goal_difference_last5",
        "home_points_per_match",
        "away_points_per_match",
    ]
    dependencies: ClassVar[List[str]] = []

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate recent form features for all matches.

        This method processes the input DataFrame chronologically, computing
        form statistics for each team based only on their previous matches.
        No future matches are considered for any calculation.

        Args:
            df (pd.DataFrame): Input dataset with required match columns.

        Returns:
            pd.DataFrame: DataFrame containing only the generated feature columns.

        Raises:
            ValueError: If required columns are missing from the input.
        """
        self.validate_input(df)
        self.logger.info(f"Generating {self.name} features for {len(df)} matches")

        # Create working copy and ensure proper data types
        working_df = df.copy()
        working_df["Date"] = pd.to_datetime(working_df["Date"])
        working_df = working_df.sort_values("Date").reset_index(drop=True)

        # Initialize feature columns
        features_df = pd.DataFrame(index=working_df.index)
        features_df[self.output_columns] = 0.0

        # Build team match history efficiently
        team_history: Dict[str, List[Dict[str, Any]]] = {}

        for idx, row in working_df.iterrows():
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]
            match_date = row["Date"]

            # Initialize team histories if needed
            if home_team not in team_history:
                team_history[home_team] = []
            if away_team not in team_history:
                team_history[away_team] = []

            # Calculate features for this match using only past matches
            home_features = self._calculate_team_features(
                team_history[home_team],
                row["FTHG"],
                row["FTAG"],
                row["FTR"],
                is_home=True,
            )
            away_features = self._calculate_team_features(
                team_history[away_team],
                row["FTAG"],
                row["FTHG"],
                row["FTR"],
                is_home=False,
            )

            # Assign features to this row
            for feature_name, value in home_features.items():
                features_df.loc[idx, f"home_{feature_name}"] = value
            for feature_name, value in away_features.items():
                features_df.loc[idx, f"away_{feature_name}"] = value

            # Add this match to team histories for future matches
            home_result = self._get_team_result(
                row["FTHG"],
                row["FTAG"],
                row["FTR"],
                is_home=True,
            )
            away_result = self._get_team_result(
                row["FTAG"],
                row["FTHG"],
                row["FTR"],
                is_home=False,
            )

            team_history[home_team].append(
                {
                    "date": match_date,
                    "points": home_result["points"],
                    "wins": home_result["wins"],
                    "draws": home_result["draws"],
                    "losses": home_result["losses"],
                    "goals_scored": row["FTHG"],
                    "goals_conceded": row["FTAG"],
                }
            )
            team_history[away_team].append(
                {
                    "date": match_date,
                    "points": away_result["points"],
                    "wins": away_result["wins"],
                    "draws": away_result["draws"],
                    "losses": away_result["losses"],
                    "goals_scored": row["FTAG"],
                    "goals_conceded": row["FTHG"],
                }
            )

        self.logger.info(f"Successfully generated {self.name} features")
        return features_df

    def _calculate_team_features(
        self,
        match_history: List[Dict[str, Any]],
        team_goals: int,
        opponent_goals: int,
        result: str,
        is_home: bool,
    ) -> Dict[str, float]:
        """
        Calculate form features for a team based on their match history.

        Uses only the last 5 matches from the history to compute rolling statistics.

        Args:
            match_history (List[Dict[str, Any]]): Team's historical matches.
            team_goals (int): Goals scored by the team in current match.
            opponent_goals (int): Goals conceded by the team in current match.
            result (str): Match result code (H, A, or D).
            is_home (bool): Whether the team is the home team.

        Returns:
            Dict[str, float]: Dictionary mapping feature names to computed values.
        """
        # Get last 5 matches
        last_five = match_history[-5:] if match_history else []

        # Initialize accumulators
        total_points = 0.0
        total_wins = 0
        total_draws = 0
        total_losses = 0
        total_goals_scored = 0.0
        total_goals_conceded = 0.0

        # Aggregate last 5 matches
        for match in last_five:
            total_points += match["points"]
            total_wins += match["wins"]
            total_draws += match["draws"]
            total_losses += match["losses"]
            total_goals_scored += match["goals_scored"]
            total_goals_conceded += match["goals_conceded"]

        # Calculate derived metrics
        form_points_last5 = float(total_points)
        goals_difference_last5 = float(total_goals_scored - total_goals_conceded)

        # Points per match (avoid division by zero)
        matches_played = len(last_five)
        points_per_match = (
            float(total_points) / matches_played if matches_played > 0 else 0.0
        )

        return {
            "form_points_last5": form_points_last5,
            "wins_last5": float(total_wins),
            "draws_last5": float(total_draws),
            "losses_last5": float(total_losses),
            "goals_scored_last5": float(total_goals_scored),
            "goals_conceded_last5": float(total_goals_conceded),
            "goal_difference_last5": goals_difference_last5,
            "points_per_match": points_per_match,
        }

    def _get_team_result(
        self,
        team_goals: int,
        opponent_goals: int,
        result: str,
        is_home: bool,
    ) -> Dict[str, int]:
        """
        Compute match outcome (points, wins, draws, losses) for a team.

        Args:
            team_goals (int): Goals scored by the team.
            opponent_goals (int): Goals conceded by the team.
            result (str): Match result code (H for home win, A for away win, D for draw).
            is_home (bool): Whether the team is the home team.

        Returns:
            Dict[str, int]: Dictionary with 'points', 'wins', 'draws', and 'losses'.
        """
        if is_home:
            if result == "H":
                return {"points": 3, "wins": 1, "draws": 0, "losses": 0}
            elif result == "D":
                return {"points": 1, "wins": 0, "draws": 1, "losses": 0}
            else:  # result == "A"
                return {"points": 0, "wins": 0, "draws": 0, "losses": 1}
        else:
            if result == "A":
                return {"points": 3, "wins": 1, "draws": 0, "losses": 0}
            elif result == "D":
                return {"points": 1, "wins": 0, "draws": 1, "losses": 0}
            else:  # result == "H"
                return {"points": 0, "wins": 0, "draws": 0, "losses": 1}
