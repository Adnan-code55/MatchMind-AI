"""
Tests for GoalDifferenceGenerator.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from backend.app.features.goal_difference import GoalDifferenceGenerator
from backend.app.features.history import TeamHistory
from backend.app.features.registry import FeatureRegistry


@pytest.fixture
def generator() -> GoalDifferenceGenerator:
    """Create a default goal difference generator."""
    return GoalDifferenceGenerator()


def test_goal_difference_generator_registers_automatically() -> None:
    """GoalDifferenceGenerator should be available through FeatureRegistry."""
    FeatureRegistry.reset()
    import backend.app.features.goal_difference as goal_difference

    importlib.reload(goal_difference)

    assert "goal_difference" in FeatureRegistry.list_generators()
    assert FeatureRegistry.get("goal_difference").name == "goal_difference"


def test_no_previous_matches_returns_zero_features(
    generator: GoalDifferenceGenerator,
) -> None:
    """The first match for both teams should have zero historical features."""
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


def test_one_previous_match_calculates_home_team_features(
    generator: GoalDifferenceGenerator,
) -> None:
    """A team's single prior match should drive its next-match features."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool"],
            "FTHG": [3, 1],
            "FTAG": [1, 0],
            "FTR": ["H", "H"],
        }
    )

    result = generator.generate(df)
    second_match = result.iloc[1]

    assert second_match["home_goals_for"] == 3.0
    assert second_match["home_goals_against"] == 1.0
    assert second_match["home_goal_difference"] == 2.0
    assert second_match["home_average_goal_difference"] == 2.0
    assert second_match["away_goal_difference"] == 0.0


def test_less_than_five_matches_uses_available_history(
    generator: GoalDifferenceGenerator,
) -> None:
    """Default window should use fewer than five matches when that is all history."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
            "AwayTeam": ["Chelsea", "Arsenal", "Spurs"],
            "FTHG": [2, 1, 4],
            "FTAG": [0, 1, 1],
            "FTR": ["H", "D", "H"],
        }
    )

    result = generator.generate(df)
    third_match = result.iloc[2]

    assert third_match["home_goals_for"] == 3.0
    assert third_match["home_goals_against"] == 1.0
    assert third_match["home_goal_difference"] == 2.0
    assert third_match["home_average_goal_difference"] == 1.0


def test_normal_operation_calculates_home_and_away_features(
    generator: GoalDifferenceGenerator,
) -> None:
    """Generator should calculate both teams' rolling goal metrics."""
    df = pd.DataFrame(
        {
            "Date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            "HomeTeam": ["Arsenal", "Liverpool", "Chelsea", "Arsenal"],
            "AwayTeam": ["Chelsea", "Arsenal", "Arsenal", "Chelsea"],
            "FTHG": [2, 1, 0, 3],
            "FTAG": [1, 1, 2, 0],
            "FTR": ["H", "D", "A", "H"],
        }
    )

    result = generator.generate(df)
    fourth_match = result.iloc[3]

    assert fourth_match["home_goals_for"] == 5.0
    assert fourth_match["home_goals_against"] == 2.0
    assert fourth_match["home_goal_difference"] == 3.0
    assert fourth_match["home_average_goal_difference"] == 1.0
    assert fourth_match["away_goals_for"] == 1.0
    assert fourth_match["away_goals_against"] == 4.0
    assert fourth_match["away_goal_difference"] == -3.0
    assert fourth_match["away_average_goal_difference"] == -1.5


def test_configurable_rolling_window_limits_history() -> None:
    """Configured window should limit TeamHistory statistics."""
    generator = GoalDifferenceGenerator(config={"window": 2})
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
            "FTHG": [5, 1, 2, 0],
            "FTAG": [0, 1, 0, 0],
            "FTR": ["H", "D", "H", "D"],
        }
    )

    result = generator.generate(df)
    fourth_match = result.iloc[3]

    assert fourth_match["home_goals_for"] == 3.0
    assert fourth_match["home_goals_against"] == 1.0
    assert fourth_match["home_goal_difference"] == 2.0
    assert fourth_match["home_average_goal_difference"] == 1.0


def test_no_future_leakage_for_current_match(
    generator: GoalDifferenceGenerator,
) -> None:
    """Current and future matches must not affect generated features."""
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
    first_match = result.iloc[0]
    second_match = result.iloc[1]

    assert first_match["home_goals_for"] == 0.0
    assert first_match["home_goal_difference"] == 0.0
    assert second_match["home_goals_for"] == 1.0
    assert second_match["home_goal_difference"] == 1.0


def test_generator_uses_team_history_get_team_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    """GoalDifferenceGenerator should delegate history calculations to TeamHistory."""
    calls = []
    original = TeamHistory.get_team_statistics

    def spy(self: TeamHistory, *args: object, **kwargs: object) -> dict[str, float]:
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TeamHistory, "get_team_statistics", spy)
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool"],
            "FTHG": [2, 1],
            "FTAG": [0, 1],
            "FTR": ["H", "D"],
        }
    )

    GoalDifferenceGenerator().generate(df)

    assert len(calls) == 4
    assert all(call[1]["limit"] == 5 for call in calls)


def test_missing_required_columns_raises_error(
    generator: GoalDifferenceGenerator,
) -> None:
    """Generator should reject input missing required match columns."""
    df = pd.DataFrame({"Date": ["2024-01-01"], "HomeTeam": ["Arsenal"]})

    with pytest.raises(ValueError):
        generator.generate(df)
