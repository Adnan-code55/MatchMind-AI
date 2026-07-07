"""
Tests for HomeAdvantageGenerator.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from backend.app.features.history import TeamHistory
from backend.app.features.home_advantage import HomeAdvantageGenerator
from backend.app.features.registry import FeatureRegistry


@pytest.fixture
def generator() -> HomeAdvantageGenerator:
    """Create a default HomeAdvantageGenerator."""
    return HomeAdvantageGenerator()


def test_home_advantage_generator_registers_automatically() -> None:
    """HomeAdvantageGenerator should register with FeatureRegistry."""
    FeatureRegistry.reset()
    import backend.app.features.home_advantage as home_advantage

    importlib.reload(home_advantage)

    assert "home_advantage" in FeatureRegistry.list_generators()
    assert FeatureRegistry.get("home_advantage").name == "home_advantage"


def test_first_match_has_zero_features(generator: HomeAdvantageGenerator) -> None:
    """Teams with no venue-specific history should receive zero features."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [1],
            "FTR": ["H"],
        }
    )

    result = generator.generate(df)

    assert set(result.columns) == set(generator.output_columns)
    assert result.iloc[0].sum() == 0.0


def test_home_only_history_uses_home_matches_only(
    generator: HomeAdvantageGenerator,
) -> None:
    """Home team venue features should ignore prior away matches."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
            "AwayTeam": ["Chelsea", "Arsenal", "Spurs"],
            "FTHG": [2, 5, 1],
            "FTAG": [0, 0, 0],
            "FTR": ["H", "H", "H"],
        }
    )

    result = generator.generate(df)
    third_match = result.iloc[2]

    assert third_match["home_home_goals_per_match"] == 2.0
    assert third_match["home_home_goals_against_per_match"] == 0.0
    assert third_match["home_home_points_per_match"] == 3.0
    assert third_match["home_home_win_rate"] == 1.0


def test_away_only_history_uses_away_matches_only(
    generator: HomeAdvantageGenerator,
) -> None:
    """Away team venue features should ignore prior home matches."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "HomeTeam": ["Chelsea", "Arsenal", "Liverpool", "Spurs"],
            "AwayTeam": ["Arsenal", "Chelsea", "Arsenal", "Arsenal"],
            "FTHG": [1, 4, 2, 0],
            "FTAG": [2, 0, 2, 1],
            "FTR": ["A", "H", "D", "A"],
        }
    )

    result = generator.generate(df)
    fourth_match = result.iloc[3]

    assert fourth_match["away_away_goals_per_match"] == 2.0
    assert fourth_match["away_away_goals_against_per_match"] == 1.5
    assert fourth_match["away_away_points_per_match"] == 2.0
    assert fourth_match["away_away_win_rate"] == 0.5
    assert fourth_match["away_away_draw_rate"] == 0.5


def test_partial_history_uses_available_venue_matches(
    generator: HomeAdvantageGenerator,
) -> None:
    """Default window should use fewer than five available venue matches."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
            "AwayTeam": ["Chelsea", "Arsenal", "Spurs"],
            "FTHG": [2, 1, 0],
            "FTAG": [0, 1, 0],
            "FTR": ["H", "D", "D"],
        }
    )

    result = generator.generate(df)
    third_match = result.iloc[2]

    assert third_match["home_home_win_rate"] == 1.0
    assert third_match["home_home_draw_rate"] == 0.0
    assert third_match["home_home_clean_sheet_rate"] == 1.0
    assert third_match["home_home_failed_to_score_rate"] == 0.0


def test_window_limits_home_history() -> None:
    """Configured window should limit home venue history."""
    generator = HomeAdvantageGenerator(config={"window": 2})
    df = pd.DataFrame(
        {
            "Date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            "HomeTeam": ["Arsenal", "Arsenal", "Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool", "Spurs", "Everton"],
            "FTHG": [5, 1, 0, 2],
            "FTAG": [0, 1, 3, 0],
            "FTR": ["H", "D", "A", "H"],
        }
    )

    result = generator.generate(df)
    fourth_match = result.iloc[3]

    assert fourth_match["home_home_goals_per_match"] == 0.5
    assert fourth_match["home_home_points_per_match"] == 0.5
    assert fourth_match["home_home_loss_rate"] == 0.5


def test_window_limits_away_history() -> None:
    """Configured window should limit away venue history."""
    generator = HomeAdvantageGenerator(config={"window": 2})
    df = pd.DataFrame(
        {
            "Date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            "HomeTeam": ["Chelsea", "Liverpool", "Spurs", "Everton"],
            "AwayTeam": ["Arsenal", "Arsenal", "Arsenal", "Arsenal"],
            "FTHG": [0, 1, 3, 0],
            "FTAG": [5, 1, 0, 2],
            "FTR": ["A", "D", "H", "A"],
        }
    )

    result = generator.generate(df)
    fourth_match = result.iloc[3]

    assert fourth_match["away_away_goals_per_match"] == 0.5
    assert fourth_match["away_away_points_per_match"] == 0.5
    assert fourth_match["away_away_loss_rate"] == 0.5


def test_no_future_leakage(generator: HomeAdvantageGenerator) -> None:
    """Current and future matches must not affect generated venue features."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "HomeTeam": ["Arsenal", "Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool", "Spurs"],
            "FTHG": [1, 10, 8],
            "FTAG": [0, 0, 0],
            "FTR": ["H", "H", "H"],
        }
    )

    result = generator.generate(df)

    assert result.iloc[0]["home_home_goals_per_match"] == 0.0
    assert result.iloc[1]["home_home_goals_per_match"] == 1.0


def test_generator_uses_team_history_with_home_and_away_venues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HomeAdvantageGenerator should delegate venue filtering to TeamHistory."""
    calls = []
    original = TeamHistory.get_team_statistics

    def spy(self: TeamHistory, *args: object, **kwargs: object) -> dict[str, float]:
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TeamHistory, "get_team_statistics", spy)
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Chelsea", "Arsenal"],
            "FTHG": [2, 1],
            "FTAG": [0, 1],
            "FTR": ["H", "D"],
        }
    )

    HomeAdvantageGenerator().generate(df)

    assert len(calls) == 4
    assert [call[1]["venue"] for call in calls] == ["home", "away", "home", "away"]
    assert all(call[1]["limit"] == 5 for call in calls)


def test_missing_required_columns_raises_error(
    generator: HomeAdvantageGenerator,
) -> None:
    """Generator should reject datasets missing required columns."""
    df = pd.DataFrame({"Date": ["2024-01-01"], "HomeTeam": ["Arsenal"]})

    with pytest.raises(ValueError):
        generator.generate(df)


def test_invalid_window_raises_error() -> None:
    """Window must be positive."""
    with pytest.raises(ValueError):
        HomeAdvantageGenerator(config={"window": 0})
