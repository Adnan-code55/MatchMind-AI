"""
Team-level historical access built on top of MatchHistory.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from ...data.logger import PipelineLogger
from .match_history import MatchHistory


class TeamHistory:
    """Provide reusable team history and aggregate statistics."""

    def __init__(self, match_history: MatchHistory) -> None:
        """Initialize team history with a match history dependency.

        Args:
            match_history: Indexed match history service.
        """
        self.match_history = match_history
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)

    def get_last_matches(
        self,
        team: str,
        before_date: Any,
        limit: Optional[int] = 5,
        **filters: Any,
    ) -> pd.DataFrame:
        """Return a team's latest matches before a date.

        Args:
            team: Club name.
            before_date: Exclusive cutoff date.
            limit: Maximum number of matches to return.
            **filters: Additional match filters.

        Returns:
            DataFrame ordered newest-first.
        """
        return self.match_history.get_matches(
            team=team,
            before_date=before_date,
            limit=limit,
            ascending=False,
            **filters,
        )

    def get_last_home_matches(
        self,
        team: str,
        before_date: Any,
        limit: Optional[int] = 5,
        **filters: Any,
    ) -> pd.DataFrame:
        """Return a team's latest home matches before a date."""
        matches = self.get_last_matches(
            team=team,
            before_date=before_date,
            limit=None,
            **filters,
        )
        filtered = matches[matches["HomeTeam"] == team]
        if limit is not None:
            filtered = filtered.head(limit)
        return filtered.copy()

    def get_last_away_matches(
        self,
        team: str,
        before_date: Any,
        limit: Optional[int] = 5,
        **filters: Any,
    ) -> pd.DataFrame:
        """Return a team's latest away matches before a date."""
        matches = self.get_last_matches(
            team=team,
            before_date=before_date,
            limit=None,
            **filters,
        )
        filtered = matches[matches["AwayTeam"] == team]
        if limit is not None:
            filtered = filtered.head(limit)
        return filtered.copy()

    def get_matches_before(
        self,
        team: str,
        before_date: Any,
        **filters: Any,
    ) -> pd.DataFrame:
        """Return all team matches strictly before a date."""
        return self.match_history.get_matches(
            team=team,
            before_date=before_date,
            ascending=True,
            **filters,
        )

    def get_team_statistics(
        self,
        team: str,
        before_date: Any,
        limit: Optional[int] = None,
        venue: str = "overall",
        **filters: Any,
    ) -> Dict[str, float]:
        """Return aggregate team statistics before a date.

        Args:
            team: Club name.
            before_date: Exclusive cutoff date.
            limit: Optional number of latest matches to include.
            venue: One of ``overall``, ``home``, or ``away``.
            **filters: Additional match filters.

        Returns:
            Dictionary of aggregate results and goal metrics.
        """
        if venue not in {"overall", "home", "away"}:
            raise ValueError("venue must be one of: overall, home, away")

        if venue == "home":
            matches = self.get_last_home_matches(team, before_date, limit, **filters)
        elif venue == "away":
            matches = self.get_last_away_matches(team, before_date, limit, **filters)
        else:
            matches = self.get_last_matches(team, before_date, limit=limit, **filters)

        required = {"FTHG", "FTAG", "FTR"}
        if not required.issubset(matches.columns):
            missing = sorted(required.difference(matches.columns))
            raise ValueError(f"Cannot calculate team statistics; missing columns: {missing}")

        stats = {
            "matches": 0.0,
            "matches_played": 0.0,
            "wins": 0.0,
            "draws": 0.0,
            "losses": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "goal_difference": 0.0,
            "points": 0.0,
            "clean_sheets": 0.0,
            "failed_to_score": 0.0,
            "goals_per_match": 0.0,
            "goals_against_per_match": 0.0,
            "points_per_match": 0.0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "clean_sheet_rate": 0.0,
            "failed_to_score_rate": 0.0,
        }

        for _, match in matches.iterrows():
            is_home = match["HomeTeam"] == team
            goals_for = float(match["FTHG"] if is_home else match["FTAG"])
            goals_against = float(match["FTAG"] if is_home else match["FTHG"])
            result = self._result_for_team(str(match["FTR"]), is_home)

            stats["matches_played"] += 1.0
            stats["goals_for"] += goals_for
            stats["goals_against"] += goals_against
            stat_key = {"win": "wins", "draw": "draws", "loss": "losses"}[result]
            stats[stat_key] += 1.0
            stats["points"] += self._points_for_result(result)
            if goals_against == 0:
                stats["clean_sheets"] += 1.0
            if goals_for == 0:
                stats["failed_to_score"] += 1.0

        if stats["matches_played"]:
            stats["matches"] = stats["matches_played"]
            stats["goal_difference"] = stats["goals_for"] - stats["goals_against"]
            stats["goals_per_match"] = stats["goals_for"] / stats["matches_played"]
            stats["goals_against_per_match"] = (
                stats["goals_against"] / stats["matches_played"]
            )
            stats["points_per_match"] = stats["points"] / stats["matches_played"]
            stats["win_rate"] = stats["wins"] / stats["matches_played"]
            stats["draw_rate"] = stats["draws"] / stats["matches_played"]
            stats["loss_rate"] = stats["losses"] / stats["matches_played"]
            stats["clean_sheet_rate"] = (
                stats["clean_sheets"] / stats["matches_played"]
            )
            stats["failed_to_score_rate"] = (
                stats["failed_to_score"] / stats["matches_played"]
            )

        return stats

    def _result_for_team(self, full_time_result: str, is_home: bool) -> str:
        """Translate full-time result into a team-specific outcome."""
        if full_time_result == "D":
            return "draw"
        if (full_time_result == "H" and is_home) or (full_time_result == "A" and not is_home):
            return "win"
        return "loss"

    def _points_for_result(self, result: str) -> float:
        """Return league points for a result label."""
        if result == "win":
            return 3.0
        if result == "draw":
            return 1.0
        return 0.0
