"""
Rest days feature generator for MatchMind AI.

This module calculates team rest intervals and schedule congestion flags using
TeamHistory as the source of historical match information.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional

import pandas as pd

from .base import FeatureGenerator
from .history import MatchHistory, TeamHistory
from .registry import FeatureRegistry


@FeatureRegistry.register
class RestDaysGenerator(FeatureGenerator):
    """Generate rest-day and schedule congestion features."""

    name: ClassVar[str] = "rest_days"
    required_columns: ClassVar[List[str]] = [
        "Date",
        "HomeTeam",
        "AwayTeam",
    ]
    output_columns: ClassVar[List[str]] = [
        "home_rest_days",
        "away_rest_days",
        "rest_day_difference",
        "home_congested_schedule",
        "away_congested_schedule",
        "long_break_flag",
    ]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        team_history: Optional[TeamHistory] = None,
    ) -> None:
        """Initialize the rest days generator.

        Args:
            config: Optional configuration. Supports ``congested_days`` and
                ``long_break_days``.
            team_history: Optional injected TeamHistory dependency.
        """
        super().__init__(config=config)
        self.congested_days = self._get_threshold("congested_days", 4)
        self.long_break_days = self._get_threshold("long_break_days", 14)
        self.team_history = team_history

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rest-day features for each match.

        Args:
            df: Input match dataset.

        Returns:
            DataFrame containing generated rest-day feature columns.

        Raises:
            ValueError: If required input columns are missing or thresholds are
                negative.
        """
        self.validate_input(df)
        self.logger.info(
            "Generating %s features for %s matches with congested_days=%s, long_break_days=%s",
            self.name,
            len(df),
            self.congested_days,
            self.long_break_days,
        )

        working_df = df.copy()
        working_df["Date"] = pd.to_datetime(working_df["Date"])
        team_history = self.team_history or TeamHistory(MatchHistory(working_df))

        features_df = pd.DataFrame(index=working_df.index)
        features_df[self.output_columns] = 0.0

        for index, row in working_df.iterrows():
            match_date = pd.Timestamp(row["Date"])
            home_rest_days = self._get_rest_days(
                team_history=team_history,
                team=str(row["HomeTeam"]),
                match_date=match_date,
            )
            away_rest_days = self._get_rest_days(
                team_history=team_history,
                team=str(row["AwayTeam"]),
                match_date=match_date,
            )

            features_df.loc[index, "home_rest_days"] = home_rest_days
            features_df.loc[index, "away_rest_days"] = away_rest_days
            features_df.loc[index, "rest_day_difference"] = (
                home_rest_days - away_rest_days
            )
            features_df.loc[index, "home_congested_schedule"] = self._is_congested(
                home_rest_days
            )
            features_df.loc[index, "away_congested_schedule"] = self._is_congested(
                away_rest_days
            )
            features_df.loc[index, "long_break_flag"] = self._has_long_break(
                home_rest_days,
                away_rest_days,
            )

        self.logger.info(f"Successfully generated {self.name} features")
        return features_df

    def _get_threshold(self, key: str, default: int) -> int:
        """Return a configured day threshold.

        Args:
            key: Configuration key to read.
            default: Default threshold value.

        Returns:
            Positive or zero integer threshold.

        Raises:
            ValueError: If the configured threshold is negative.
        """
        threshold = int(self.config.get(key, default))
        if threshold < 0:
            raise ValueError(f"{key} must be greater than or equal to 0")
        return threshold

    def _get_rest_days(
        self,
        team_history: TeamHistory,
        team: str,
        match_date: pd.Timestamp,
    ) -> float:
        """Return days since a team's most recent previous match.

        Args:
            team_history: TeamHistory dependency.
            team: Team name to query.
            match_date: Current match date.

        Returns:
            Number of rest days, or 0.0 when no previous match exists.
        """
        last_matches = team_history.get_last_matches(
            team=team,
            before_date=match_date,
            limit=1,
        )
        if last_matches.empty:
            return 0.0

        previous_date = pd.Timestamp(last_matches.iloc[0]["Date"])
        return float((match_date - previous_date).days)

    def _is_congested(self, rest_days: float) -> float:
        """Return whether a rest interval is congested."""
        if rest_days <= 0:
            return 0.0
        return float(rest_days <= self.congested_days)

    def _has_long_break(self, home_rest_days: float, away_rest_days: float) -> float:
        """Return whether either team has a long break."""
        return float(
            home_rest_days >= self.long_break_days
            or away_rest_days >= self.long_break_days
        )
