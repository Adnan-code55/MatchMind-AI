"""
Chronological match access for MatchMind AI historical data.

This module owns reusable match-level indexing and filtering so feature
generators do not scan historical DataFrames themselves.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Set

import pandas as pd

from ...data.logger import PipelineLogger
from .cache import HistoryCache


class MatchHistory:
    """Provide indexed, chronological access to historical matches."""

    required_columns = ["Date", "HomeTeam", "AwayTeam"]

    def __init__(
        self,
        matches: pd.DataFrame,
        cache: Optional[HistoryCache] = None,
        date_column: str = "Date",
    ) -> None:
        """Initialize indexed match history.

        Args:
            matches: Match dataset to index.
            cache: Optional cache dependency for repeated queries.
            date_column: Column containing match dates.

        Raises:
            ValueError: If required columns are missing.
        """
        self.date_column = date_column
        self.cache = cache or HistoryCache()
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)
        self._matches = self._prepare_matches(matches)
        self._dates = self._matches[self.date_column]
        self._team_index = self._build_team_index()
        self._column_indexes = self._build_column_indexes(
            ["league", "League", "season", "Season", "competition", "Competition", "country", "Country"]
        )
        self.logger.info(f"Indexed {len(self._matches)} matches for historical access.")

    @property
    def matches(self) -> pd.DataFrame:
        """Return a defensive copy of chronological matches."""
        return self._matches.copy()

    def get_matches(
        self,
        team: Optional[str] = None,
        league: Optional[Any] = None,
        season: Optional[Any] = None,
        competition: Optional[Any] = None,
        country: Optional[Any] = None,
        date: Optional[Any] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        before_date: Optional[Any] = None,
        limit: Optional[int] = None,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Return matches filtered by indexed dimensions in chronological order.

        Args:
            team: Optional club appearing as home or away team.
            league: Optional league filter.
            season: Optional season filter.
            competition: Optional competition filter.
            country: Optional country filter.
            date: Optional exact match date filter.
            start_date: Optional inclusive lower date bound.
            end_date: Optional inclusive upper date bound.
            before_date: Optional exclusive upper date bound for historical queries.
            limit: Optional maximum number of rows to return.
            ascending: Whether to return oldest-first order.

        Returns:
            DataFrame containing matching rows.
        """
        self._validate_limit(limit)
        cache_key = self.cache.make_key(
            "matches",
            team=team,
            league=league,
            season=season,
            competition=competition,
            country=country,
            date=self._date_key(date),
            start_date=self._date_key(start_date),
            end_date=self._date_key(end_date),
            before_date=self._date_key(before_date),
            limit=limit,
            ascending=ascending,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        candidate_indexes: Optional[Set[int]] = None
        candidate_indexes = self._intersect(candidate_indexes, self._indexes_for_team(team))
        for column_name, value in self._filter_columns(
            league=league,
            season=season,
            competition=competition,
            country=country,
        ).items():
            candidate_indexes = self._intersect(
                candidate_indexes,
                self._indexes_for_column_value(column_name, value),
            )

        date_indexes = self._indexes_for_dates(date, start_date, end_date, before_date)
        candidate_indexes = self._intersect(candidate_indexes, date_indexes)

        if candidate_indexes is None:
            indexes = list(range(len(self._matches)))
        else:
            indexes = sorted(candidate_indexes)

        if not ascending:
            indexes = list(reversed(indexes))
        if limit is not None:
            indexes = indexes[:limit]

        result = self._matches.iloc[indexes].copy()
        self.cache.put(cache_key, result)
        return result

    def get_matches_before(
        self,
        before_date: Any,
        **filters: Any,
    ) -> pd.DataFrame:
        """Return matches strictly before a given date.

        Args:
            before_date: Exclusive historical cutoff date.
            **filters: Additional filters accepted by ``get_matches``.

        Returns:
            DataFrame containing historical matches only.
        """
        return self.get_matches(before_date=before_date, **filters)

    def _prepare_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Validate and sort match data for indexed access."""
        missing = [column for column in self.required_columns if column not in matches.columns]
        if missing:
            raise ValueError(f"Match history is missing required columns: {missing}")
        if self.date_column not in matches.columns:
            raise ValueError(f"Match history is missing date column: {self.date_column}")

        prepared = matches.copy()
        prepared[self.date_column] = pd.to_datetime(prepared[self.date_column])
        return prepared.sort_values(self.date_column, kind="mergesort").reset_index(drop=True)

    def _build_team_index(self) -> Dict[str, Set[int]]:
        """Build home-or-away team indexes."""
        index: Dict[str, Set[int]] = defaultdict(set)
        for row_number, row in self._matches[["HomeTeam", "AwayTeam"]].iterrows():
            index[str(row["HomeTeam"])].add(row_number)
            index[str(row["AwayTeam"])].add(row_number)
        return dict(index)

    def _build_column_indexes(self, columns: Sequence[str]) -> Dict[str, Dict[Any, Set[int]]]:
        """Build reusable indexes for optional categorical filters."""
        indexes: Dict[str, Dict[Any, Set[int]]] = {}
        for column in columns:
            if column not in self._matches.columns:
                continue
            column_index: Dict[Any, Set[int]] = defaultdict(set)
            for row_number, value in self._matches[column].items():
                column_index[value].add(row_number)
            indexes[column] = dict(column_index)
        return indexes

    def _filter_columns(self, **filters: Any) -> Dict[str, Any]:
        """Resolve public filter names to dataset columns."""
        resolved: Dict[str, Any] = {}
        for name, value in filters.items():
            if value is None:
                continue
            for candidate in (name, name.capitalize()):
                if candidate in self._column_indexes:
                    resolved[candidate] = value
                    break
        return resolved

    def _indexes_for_team(self, team: Optional[str]) -> Optional[Set[int]]:
        """Return indexed rows for a team, if provided."""
        if team is None:
            return None
        return set(self._team_index.get(str(team), set()))

    def _indexes_for_column_value(self, column: str, value: Any) -> Set[int]:
        """Return indexed rows for a categorical column value."""
        return set(self._column_indexes.get(column, {}).get(value, set()))

    def _indexes_for_dates(
        self,
        date: Optional[Any],
        start_date: Optional[Any],
        end_date: Optional[Any],
        before_date: Optional[Any],
    ) -> Optional[Set[int]]:
        """Return row indexes matching date bounds."""
        if date is not None:
            target = self._to_timestamp(date)
            start_pos = int(self._dates.searchsorted(target, side="left"))
            end_pos = int(self._dates.searchsorted(target, side="right"))
            return set(range(start_pos, end_pos))

        lower = 0
        upper = len(self._matches)
        if start_date is not None:
            lower = int(self._dates.searchsorted(self._to_timestamp(start_date), side="left"))
        if end_date is not None:
            upper = min(
                upper,
                int(self._dates.searchsorted(self._to_timestamp(end_date), side="right")),
            )
        if before_date is not None:
            upper = min(
                upper,
                int(self._dates.searchsorted(self._to_timestamp(before_date), side="left")),
            )

        if lower == 0 and upper == len(self._matches):
            return None
        return set(range(lower, upper))

    def _intersect(
        self,
        current: Optional[Set[int]],
        next_indexes: Optional[Set[int]],
    ) -> Optional[Set[int]]:
        """Intersect index sets while allowing unconstrained filters."""
        if next_indexes is None:
            return current
        if current is None:
            return set(next_indexes)
        return current.intersection(next_indexes)

    def _to_timestamp(self, value: Any) -> pd.Timestamp:
        """Convert a date-like value into a pandas Timestamp."""
        return pd.Timestamp(value)

    def _date_key(self, value: Optional[Any]) -> Optional[str]:
        """Normalize date-like cache key values."""
        if value is None:
            return None
        return self._to_timestamp(value).isoformat()

    def _validate_limit(self, limit: Optional[int]) -> None:
        """Validate optional limit input."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be greater than or equal to 0")
