"""
Head-to-head historical access between two clubs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from ...data.logger import PipelineLogger
from .match_history import MatchHistory


class HeadToHeadHistory:
    """Provide previous meetings between two clubs."""

    def __init__(self, match_history: MatchHistory) -> None:
        """Initialize head-to-head history with a match history dependency.

        Args:
            match_history: Indexed match history service.
        """
        self.match_history = match_history
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)

    def get_previous_meetings(
        self,
        home_team: str,
        away_team: str,
        before_date: Any,
        limit: Optional[int] = None,
        mode: str = "overall",
        **filters: Any,
    ) -> pd.DataFrame:
        """Return previous meetings between two clubs.

        Args:
            home_team: Home-side club in the fixture being evaluated.
            away_team: Away-side club in the fixture being evaluated.
            before_date: Exclusive cutoff date.
            limit: Optional maximum number of meetings to return.
            mode: ``overall``, ``home-only``, or ``away-only``.
            **filters: Additional match filters.

        Returns:
            DataFrame ordered newest-first.
        """
        if mode not in {"overall", "home-only", "away-only"}:
            raise ValueError("mode must be one of: overall, home-only, away-only")

        base_matches = self.match_history.get_matches(
            team=home_team,
            before_date=before_date,
            ascending=False,
            **filters,
        )

        if mode == "home-only":
            mask = (base_matches["HomeTeam"] == home_team) & (
                base_matches["AwayTeam"] == away_team
            )
        elif mode == "away-only":
            mask = (base_matches["HomeTeam"] == away_team) & (
                base_matches["AwayTeam"] == home_team
            )
        else:
            mask = (
                ((base_matches["HomeTeam"] == home_team) & (base_matches["AwayTeam"] == away_team))
                | ((base_matches["HomeTeam"] == away_team) & (base_matches["AwayTeam"] == home_team))
            )

        meetings = base_matches[mask].copy()
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be greater than or equal to 0")
            meetings = meetings.head(limit)
        return meetings

    def get_overall(
        self,
        team_a: str,
        team_b: str,
        before_date: Any,
        limit: Optional[int] = None,
        **filters: Any,
    ) -> pd.DataFrame:
        """Return all previous meetings between two clubs."""
        return self.get_previous_meetings(
            team_a,
            team_b,
            before_date,
            limit=limit,
            mode="overall",
            **filters,
        )

    def get_head_to_head_statistics(
        self,
        home_team: str,
        away_team: str,
        before_date: Any,
        limit: Optional[int] = 5,
        **filters: Any,
    ) -> Dict[str, float]:
        """Return aggregate head-to-head statistics before a fixture.

        Statistics are oriented to the current fixture: ``home_wins`` and
        ``home_goals`` refer to ``home_team`` even when that team was away in a
        previous meeting.

        Args:
            home_team: Home-side club in the fixture being evaluated.
            away_team: Away-side club in the fixture being evaluated.
            before_date: Exclusive cutoff date.
            limit: Optional maximum number of previous meetings to include.
            **filters: Additional match filters.

        Returns:
            Dictionary of aggregate head-to-head metrics.
        """
        meetings = self.get_previous_meetings(
            home_team=home_team,
            away_team=away_team,
            before_date=before_date,
            limit=limit,
            mode="overall",
            **filters,
        )

        required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not required.issubset(meetings.columns):
            missing = sorted(required.difference(meetings.columns))
            raise ValueError(
                f"Cannot calculate head-to-head statistics; missing columns: {missing}"
            )

        stats = {
            "matches": 0.0,
            "home_wins": 0.0,
            "away_wins": 0.0,
            "draws": 0.0,
            "home_goals": 0.0,
            "away_goals": 0.0,
            "average_goals": 0.0,
            "goal_difference": 0.0,
        }

        for _, match in meetings.iterrows():
            current_home_was_home = match["HomeTeam"] == home_team
            home_goals = float(match["FTHG"] if current_home_was_home else match["FTAG"])
            away_goals = float(match["FTAG"] if current_home_was_home else match["FTHG"])

            stats["matches"] += 1.0
            stats["home_goals"] += home_goals
            stats["away_goals"] += away_goals

            if home_goals > away_goals:
                stats["home_wins"] += 1.0
            elif away_goals > home_goals:
                stats["away_wins"] += 1.0
            else:
                stats["draws"] += 1.0

        if stats["matches"]:
            total_goals = stats["home_goals"] + stats["away_goals"]
            stats["average_goals"] = total_goals / stats["matches"]
            stats["goal_difference"] = stats["home_goals"] - stats["away_goals"]

        return stats
