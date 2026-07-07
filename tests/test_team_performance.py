"""
Tests for TeamPerformanceGenerator.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from backend.app.features.history import MatchHistory, TeamHistory
from backend.app.features.registry import FeatureRegistry
from backend.app.features.team_performance import TeamPerformanceGenerator


@pytest.fixture
def generator() -> TeamPerformanceGenerator:
    """Create a default TeamPerformanceGenerator."""
    return TeamPerformanceGenerator()


def test_team_performance_generator_registers_automatically() -> None:
    """TeamPerformanceGenerator should register with FeatureRegistry."""
    FeatureRegistry.reset()
    import backend.app.features.team_performance as team_performance

    importlib.reload(team_performance)

    assert "team_performance" in FeatureRegistry.list_generators()
    assert FeatureRegistry.get("team_performance").name == "team_performance"


def test_team_history_returns_extended_statistics() -> None:
    """TeamHistory should expose milestone 2.4 aggregate fields."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Chelsea", "Arsenal"],
            "FTHG": [2, 0],
            "FTAG": [0, 0],
            "FTR": ["H", "D"],
        }
    )
    team_history = TeamHistory(MatchHistory(df))

    stats = team_history.get_team_statistics("Arsenal", before_date="2024-01-03")

    assert stats["matches"] == 2.0
    assert stats["clean_sheets"] == 2.0
    assert stats["failed_to_score"] == 1.0


def test_first_match_has_zero_features(generator: TeamPerformanceGenerator) -> None:
    """Teams with no previous matches should receive zero rates."""
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


def test_partial_history_uses_available_matches(
    generator: TeamPerformanceGenerator,
) -> None:
    """A team with fewer than five matches should use available history."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
            "AwayTeam": ["Chelsea", "Arsenal", "Spurs"],
            "FTHG": [2, 1, 3],
            "FTAG": [0, 1, 0],
            "FTR": ["H", "D", "H"],
        }
    )

    result = generator.generate(df)
    third_match = result.iloc[2]

    assert third_match["home_goals_per_match"] == 1.5
    assert third_match["home_goals_against_per_match"] == 0.5
    assert third_match["home_points_per_match"] == 2.0
    assert third_match["home_win_rate"] == 0.5
    assert third_match["home_draw_rate"] == 0.5
    assert third_match["home_loss_rate"] == 0.0


def test_normal_history_calculates_home_and_away_features(
    generator: TeamPerformanceGenerator,
) -> None:
    """Generator should calculate home and away performance from history."""
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

    assert fourth_match["home_goals_per_match"] == pytest.approx(5.0 / 3.0)
    assert fourth_match["home_goals_against_per_match"] == pytest.approx(2.0 / 3.0)
    assert fourth_match["home_points_per_match"] == pytest.approx(7.0 / 3.0)
    assert fourth_match["away_goals_per_match"] == 0.5
    assert fourth_match["away_goals_against_per_match"] == 2.0
    assert fourth_match["away_points_per_match"] == 0.0


def test_clean_sheet_rate_calculation(generator: TeamPerformanceGenerator) -> None:
    """Clean sheet rate should count prior matches with zero goals conceded."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "HomeTeam": ["Arsenal", "Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool", "Spurs"],
            "FTHG": [2, 1, 0],
            "FTAG": [0, 1, 0],
            "FTR": ["H", "D", "D"],
        }
    )

    result = generator.generate(df)

    assert result.iloc[2]["home_clean_sheet_rate"] == 0.5


def test_failed_to_score_rate_calculation(
    generator: TeamPerformanceGenerator,
) -> None:
    """Failed-to-score rate should count prior matches with zero goals for."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "HomeTeam": ["Arsenal", "Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool", "Spurs"],
            "FTHG": [0, 2, 1],
            "FTAG": [1, 0, 0],
            "FTR": ["A", "H", "H"],
        }
    )

    result = generator.generate(df)

    assert result.iloc[2]["home_failed_to_score_rate"] == 0.5


def test_win_draw_loss_rates(generator: TeamPerformanceGenerator) -> None:
    """Win, draw, and loss rates should be calculated from prior history."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "HomeTeam": ["Arsenal", "Arsenal", "Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool", "Spurs", "Everton"],
            "FTHG": [2, 1, 0, 1],
            "FTAG": [0, 1, 3, 0],
            "FTR": ["H", "D", "A", "H"],
        }
    )

    result = generator.generate(df)
    fourth_match = result.iloc[3]

    assert fourth_match["home_win_rate"] == pytest.approx(1.0 / 3.0)
    assert fourth_match["home_draw_rate"] == pytest.approx(1.0 / 3.0)
    assert fourth_match["home_loss_rate"] == pytest.approx(1.0 / 3.0)


def test_points_per_match(generator: TeamPerformanceGenerator) -> None:
    """Points per match should average prior match points."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "HomeTeam": ["Arsenal", "Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool", "Spurs"],
            "FTHG": [2, 1, 0],
            "FTAG": [0, 1, 3],
            "FTR": ["H", "D", "A"],
        }
    )

    result = generator.generate(df)

    assert result.iloc[2]["home_points_per_match"] == 2.0


def test_configurable_window_limits_history() -> None:
    """Configured rolling window should limit TeamHistory statistics."""
    generator = TeamPerformanceGenerator(config={"window": 2})
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "HomeTeam": ["Arsenal", "Arsenal", "Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool", "Spurs", "Everton"],
            "FTHG": [5, 1, 0, 2],
            "FTAG": [0, 1, 3, 0],
            "FTR": ["H", "D", "A", "H"],
        }
    )

    result = generator.generate(df)

    assert result.iloc[3]["home_goals_per_match"] == 0.5
    assert result.iloc[3]["home_points_per_match"] == 0.5


def test_no_future_leakage(generator: TeamPerformanceGenerator) -> None:
    """Current and future matches should not affect feature values."""
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

    assert result.iloc[0]["home_goals_per_match"] == 0.0
    assert result.iloc[1]["home_goals_per_match"] == 1.0


def test_missing_required_columns_raises_error(
    generator: TeamPerformanceGenerator,
) -> None:
    """Generator should reject datasets missing required columns."""
    df = pd.DataFrame({"Date": ["2024-01-01"], "HomeTeam": ["Arsenal"]})

    with pytest.raises(ValueError):
        generator.generate(df)
