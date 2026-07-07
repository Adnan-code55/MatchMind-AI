"""
Tests for the MatchMind AI History Engine.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.app.features.history import (
    HeadToHeadHistory,
    HistoryCache,
    MatchHistory,
    TeamHistory,
)


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """Create reusable multi-season match history."""
    return pd.DataFrame(
        {
            "Date": [
                "2024-01-06",
                "2024-01-01",
                "2024-01-10",
                "2024-02-01",
                "2024-02-15",
                "2024-03-01",
                "2024-03-10",
            ],
            "HomeTeam": [
                "Chelsea",
                "Arsenal",
                "Liverpool",
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Spurs",
            ],
            "AwayTeam": [
                "Arsenal",
                "Chelsea",
                "Arsenal",
                "Liverpool",
                "Arsenal",
                "Chelsea",
                "Arsenal",
            ],
            "FTHG": [1, 2, 0, 3, 2, 1, 0],
            "FTAG": [2, 1, 1, 1, 2, 0, 2],
            "FTR": ["A", "H", "A", "H", "D", "H", "A"],
            "League": ["EPL", "EPL", "EPL", "EPL", "EPL", "Cup", "EPL"],
            "Season": ["2023/24", "2023/24", "2023/24", "2023/24", "2023/24", "2023/24", "2023/24"],
            "Competition": ["League", "League", "League", "League", "League", "FA Cup", "League"],
            "Country": ["England"] * 7,
        }
    )


@pytest.fixture
def match_history(sample_matches: pd.DataFrame) -> MatchHistory:
    """Create indexed match history."""
    return MatchHistory(sample_matches, cache=HistoryCache(max_size=20))


def test_match_history_returns_chronological_matches(match_history: MatchHistory) -> None:
    """MatchHistory should sort unsorted input by date."""
    result = match_history.get_matches()

    assert result["Date"].tolist() == sorted(result["Date"].tolist())


def test_match_history_filters_by_team_and_metadata(match_history: MatchHistory) -> None:
    """MatchHistory should combine team and indexed metadata filters."""
    result = match_history.get_matches(team="Arsenal", league="Cup")

    assert len(result) == 1
    assert result.iloc[0]["Competition"] == "FA Cup"
    assert result.iloc[0]["HomeTeam"] == "Arsenal"


def test_match_history_filters_exact_date(match_history: MatchHistory) -> None:
    """MatchHistory should support exact date filtering."""
    result = match_history.get_matches(date="2024-02-15")

    assert len(result) == 1
    assert result.iloc[0]["HomeTeam"] == "Chelsea"


def test_match_history_before_date_excludes_current_and_future(
    match_history: MatchHistory,
) -> None:
    """Historical cutoff should be exclusive."""
    result = match_history.get_matches(team="Arsenal", before_date="2024-02-01")

    assert result["Date"].max() < pd.Timestamp("2024-02-01")
    assert "2024-02-01" not in result["Date"].astype(str).tolist()


def test_team_history_last_matches_are_newest_first(match_history: MatchHistory) -> None:
    """TeamHistory should return latest historical matches first."""
    team_history = TeamHistory(match_history)

    result = team_history.get_last_matches("Arsenal", before_date="2024-03-01", limit=3)

    assert result["Date"].tolist() == [
        pd.Timestamp("2024-02-15"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-01-10"),
    ]


def test_team_history_home_and_away_filters(match_history: MatchHistory) -> None:
    """TeamHistory should support home-only and away-only latest matches."""
    team_history = TeamHistory(match_history)

    home = team_history.get_last_home_matches("Arsenal", before_date="2024-03-20", limit=2)
    away = team_history.get_last_away_matches("Arsenal", before_date="2024-03-20", limit=2)

    assert home["HomeTeam"].tolist() == ["Arsenal", "Arsenal"]
    assert away["AwayTeam"].tolist() == ["Arsenal", "Arsenal"]


def test_team_statistics_use_only_history_before_date(match_history: MatchHistory) -> None:
    """Team statistics should not include current or future matches."""
    team_history = TeamHistory(match_history)

    stats = team_history.get_team_statistics("Arsenal", before_date="2024-02-15")

    assert stats["matches_played"] == 4.0
    assert stats["wins"] == 4.0
    assert stats["points"] == 12.0
    assert stats["goals_for"] == 8.0
    assert stats["goals_against"] == 3.0
    assert stats["points_per_match"] == 3.0


def test_head_to_head_overall_excludes_future(match_history: MatchHistory) -> None:
    """Head-to-head history should include only previous meetings."""
    h2h = HeadToHeadHistory(match_history)

    result = h2h.get_previous_meetings(
        "Arsenal",
        "Chelsea",
        before_date="2024-03-01",
        mode="overall",
    )

    assert result["Date"].tolist() == [
        pd.Timestamp("2024-02-15"),
        pd.Timestamp("2024-01-06"),
        pd.Timestamp("2024-01-01"),
    ]
    assert pd.Timestamp("2024-03-01") not in result["Date"].tolist()


def test_head_to_head_home_and_away_modes(match_history: MatchHistory) -> None:
    """Head-to-head history should support fixture-specific venue modes."""
    h2h = HeadToHeadHistory(match_history)

    home_only = h2h.get_previous_meetings(
        "Arsenal",
        "Chelsea",
        before_date="2024-04-01",
        mode="home-only",
    )
    away_only = h2h.get_previous_meetings(
        "Arsenal",
        "Chelsea",
        before_date="2024-04-01",
        mode="away-only",
    )

    assert home_only["HomeTeam"].tolist() == ["Arsenal", "Arsenal"]
    assert home_only["AwayTeam"].tolist() == ["Chelsea", "Chelsea"]
    assert away_only["HomeTeam"].tolist() == ["Chelsea", "Chelsea"]
    assert away_only["AwayTeam"].tolist() == ["Arsenal", "Arsenal"]


def test_head_to_head_limit(match_history: MatchHistory) -> None:
    """Head-to-head history should apply limits after newest-first ordering."""
    h2h = HeadToHeadHistory(match_history)

    result = h2h.get_previous_meetings(
        "Arsenal",
        "Chelsea",
        before_date="2024-04-01",
        limit=2,
    )

    assert len(result) == 2
    assert result["Date"].tolist() == [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-02-15")]


def test_history_cache_returns_defensive_dataframe_copies() -> None:
    """Cache should be transparent and protect cached DataFrames from mutation."""
    cache = HistoryCache(max_size=2)
    original = pd.DataFrame({"value": [1]})

    cache.put("frame", original)
    first = cache.get("frame")
    assert first is not None
    first.loc[0, "value"] = 99
    second = cache.get("frame")

    assert second is not None
    assert second.loc[0, "value"] == 1


def test_history_cache_evicts_least_recently_used_entry() -> None:
    """Cache should keep the most recently used entries within max size."""
    cache = HistoryCache(max_size=2)

    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_unknown_team_returns_empty_history(match_history: MatchHistory) -> None:
    """Unknown teams should return an empty DataFrame with expected columns."""
    result = match_history.get_matches(team="Unknown FC", before_date="2024-04-01")

    assert result.empty
    assert "HomeTeam" in result.columns


def test_invalid_limit_raises_error(match_history: MatchHistory) -> None:
    """Negative limits should be rejected."""
    with pytest.raises(ValueError):
        match_history.get_matches(limit=-1)
