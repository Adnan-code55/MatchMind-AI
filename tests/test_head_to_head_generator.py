"""
Tests for HeadToHeadGenerator.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from backend.app.features.head_to_head_generator import HeadToHeadGenerator
from backend.app.features.history import HeadToHeadHistory, MatchHistory
from backend.app.features.registry import FeatureRegistry


@pytest.fixture
def generator() -> HeadToHeadGenerator:
    """Create a default HeadToHeadGenerator."""
    return HeadToHeadGenerator()


def test_head_to_head_generator_registers_automatically() -> None:
    """HeadToHeadGenerator should register with FeatureRegistry."""
    FeatureRegistry.reset()
    import backend.app.features.head_to_head_generator as head_to_head_generator

    importlib.reload(head_to_head_generator)

    assert "head_to_head" in FeatureRegistry.list_generators()
    assert FeatureRegistry.get("head_to_head").name == "head_to_head"


def test_no_previous_meetings_returns_zero_features(
    generator: HeadToHeadGenerator,
) -> None:
    """Fixtures with no prior meetings should receive zero features."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [1],
        }
    )

    result = generator.generate(df)

    assert set(result.columns) == set(generator.output_columns)
    assert result.iloc[0].sum() == 0.0


def test_one_previous_meeting(generator: HeadToHeadGenerator) -> None:
    """A single prior meeting should populate H2H counts and rates."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-10"],
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Chelsea"],
            "FTHG": [2, 1],
            "FTAG": [0, 0],
        }
    )

    result = generator.generate(df)
    second_match = result.iloc[1]

    assert second_match["home_h2h_wins"] == 1.0
    assert second_match["away_h2h_wins"] == 0.0
    assert second_match["h2h_draws"] == 0.0
    assert second_match["home_h2h_win_rate"] == 1.0
    assert second_match["h2h_matches_played"] == 1.0


def test_multiple_meetings_orient_to_current_fixture(
    generator: HeadToHeadGenerator,
) -> None:
    """Stats should be oriented to the current home and away teams."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Arsenal", "Chelsea", "Chelsea"],
            "FTHG": [2, 3, 1, 0],
            "FTAG": [0, 1, 1, 0],
        }
    )

    result = generator.generate(df)
    fourth_match = result.iloc[3]

    assert fourth_match["home_h2h_wins"] == 1.0
    assert fourth_match["away_h2h_wins"] == 1.0
    assert fourth_match["h2h_draws"] == 1.0
    assert fourth_match["home_h2h_goals"] == 4.0
    assert fourth_match["away_h2h_goals"] == 4.0
    assert fourth_match["h2h_matches_played"] == 3.0


def test_window_limit() -> None:
    """Configured rolling window should limit previous meetings."""
    generator = HeadToHeadGenerator(config={"window": 2})
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
            "HomeTeam": ["Arsenal", "Arsenal", "Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Chelsea", "Chelsea", "Chelsea"],
            "FTHG": [5, 0, 1, 0],
            "FTAG": [0, 2, 1, 0],
        }
    )

    result = generator.generate(df)
    fourth_match = result.iloc[3]

    assert fourth_match["h2h_matches_played"] == 2.0
    assert fourth_match["home_h2h_wins"] == 0.0
    assert fourth_match["away_h2h_wins"] == 1.0
    assert fourth_match["h2h_draws"] == 1.0
    assert fourth_match["home_h2h_goals"] == 1.0
    assert fourth_match["away_h2h_goals"] == 3.0


def test_future_leakage_is_prevented(generator: HeadToHeadGenerator) -> None:
    """Future meetings must not affect earlier H2H features."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-10", "2024-01-01", "2024-01-20"],
            "HomeTeam": ["Arsenal", "Arsenal", "Chelsea"],
            "AwayTeam": ["Chelsea", "Chelsea", "Arsenal"],
            "FTHG": [9, 1, 0],
            "FTAG": [0, 0, 0],
        }
    )

    result = generator.generate(df)
    first_chronological_fixture = result.iloc[1]
    jan_tenth_fixture = result.iloc[0]

    assert first_chronological_fixture["h2h_matches_played"] == 0.0
    assert jan_tenth_fixture["h2h_matches_played"] == 1.0
    assert jan_tenth_fixture["home_h2h_goals"] == 1.0


def test_goal_calculations(generator: HeadToHeadGenerator) -> None:
    """Goal totals, averages, and differences should be correct."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-08", "2024-01-15"],
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
            "AwayTeam": ["Chelsea", "Arsenal", "Chelsea"],
            "FTHG": [3, 2, 0],
            "FTAG": [1, 2, 0],
        }
    )

    result = generator.generate(df)
    third_match = result.iloc[2]

    assert third_match["home_h2h_goals"] == 5.0
    assert third_match["away_h2h_goals"] == 3.0
    assert third_match["h2h_average_goals"] == 4.0
    assert third_match["recent_h2h_goal_difference"] == 2.0


def test_win_rates(generator: HeadToHeadGenerator) -> None:
    """Win and draw rates should be derived from H2H counts."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"],
            "HomeTeam": ["Arsenal", "Arsenal", "Chelsea", "Arsenal"],
            "AwayTeam": ["Chelsea", "Chelsea", "Arsenal", "Chelsea"],
            "FTHG": [2, 1, 2, 0],
            "FTAG": [0, 1, 0, 0],
        }
    )

    result = generator.generate(df)
    fourth_match = result.iloc[3]

    assert fourth_match["home_h2h_win_rate"] == pytest.approx(1.0 / 3.0)
    assert fourth_match["away_h2h_win_rate"] == pytest.approx(1.0 / 3.0)
    assert fourth_match["h2h_draw_rate"] == pytest.approx(1.0 / 3.0)


def test_head_to_head_history_statistics() -> None:
    """HeadToHeadHistory should expose aggregate H2H statistics."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-08", "2024-01-15"],
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
            "AwayTeam": ["Chelsea", "Arsenal", "Chelsea"],
            "FTHG": [1, 0, 0],
            "FTAG": [0, 2, 0],
        }
    )
    h2h = HeadToHeadHistory(MatchHistory(df))

    stats = h2h.get_head_to_head_statistics(
        "Arsenal",
        "Chelsea",
        before_date="2024-01-15",
    )

    assert stats["matches"] == 2.0
    assert stats["home_wins"] == 2.0
    assert stats["home_goals"] == 3.0
    assert stats["away_goals"] == 0.0


def test_generator_uses_head_to_head_history_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generator should delegate historical aggregation to HeadToHeadHistory."""
    calls = []
    original = HeadToHeadHistory.get_head_to_head_statistics

    def spy(
        self: HeadToHeadHistory,
        *args: object,
        **kwargs: object,
    ) -> dict[str, float]:
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HeadToHeadHistory, "get_head_to_head_statistics", spy)
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-08"],
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Chelsea"],
            "FTHG": [1, 0],
            "FTAG": [0, 0],
        }
    )

    HeadToHeadGenerator().generate(df)

    assert len(calls) == 2
    assert all(call[1]["limit"] == 5 for call in calls)


def test_missing_required_columns_raises_error(
    generator: HeadToHeadGenerator,
) -> None:
    """Generator should reject datasets missing required columns."""
    df = pd.DataFrame({"Date": ["2024-01-01"], "HomeTeam": ["Arsenal"]})

    with pytest.raises(ValueError):
        generator.generate(df)


def test_invalid_window_raises_error() -> None:
    """Window must be positive."""
    with pytest.raises(ValueError):
        HeadToHeadGenerator(config={"window": 0})
