"""
Tests for RestDaysGenerator.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from backend.app.features.history import TeamHistory
from backend.app.features.registry import FeatureRegistry
from backend.app.features.rest_days import RestDaysGenerator


@pytest.fixture
def generator() -> RestDaysGenerator:
    """Create a default RestDaysGenerator."""
    return RestDaysGenerator()


def test_rest_days_generator_registers_automatically() -> None:
    """RestDaysGenerator should register with FeatureRegistry."""
    FeatureRegistry.reset()
    import backend.app.features.rest_days as rest_days

    importlib.reload(rest_days)

    assert "rest_days" in FeatureRegistry.list_generators()
    assert FeatureRegistry.get("rest_days").name == "rest_days"


def test_first_match_has_zero_rest_features(generator: RestDaysGenerator) -> None:
    """Teams with no previous matches should receive zero rest features."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
        }
    )

    result = generator.generate(df)

    assert set(result.columns) == set(generator.output_columns)
    assert result.iloc[0].sum() == 0.0


def test_no_previous_matches_for_one_team(generator: RestDaysGenerator) -> None:
    """A team with no prior match should receive zero rest days."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-08"],
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool"],
        }
    )

    result = generator.generate(df)
    second_match = result.iloc[1]

    assert second_match["home_rest_days"] == 7.0
    assert second_match["away_rest_days"] == 0.0
    assert second_match["rest_day_difference"] == 7.0


def test_normal_rest_intervals(generator: RestDaysGenerator) -> None:
    """Rest days should be calculated from each team's latest prior match."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-03", "2024-01-08"],
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
            "AwayTeam": ["Chelsea", "Arsenal", "Chelsea"],
        }
    )

    result = generator.generate(df)
    third_match = result.iloc[2]

    assert third_match["home_rest_days"] == 5.0
    assert third_match["away_rest_days"] == 5.0
    assert third_match["rest_day_difference"] == 0.0
    assert third_match["home_congested_schedule"] == 0.0
    assert third_match["away_congested_schedule"] == 0.0


def test_congested_schedules(generator: RestDaysGenerator) -> None:
    """Rest intervals at or below the congestion threshold should be flagged."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-03", "2024-01-05"],
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
            "AwayTeam": ["Chelsea", "Arsenal", "Chelsea"],
        }
    )

    result = generator.generate(df)
    third_match = result.iloc[2]

    assert third_match["home_rest_days"] == 2.0
    assert third_match["away_rest_days"] == 2.0
    assert third_match["home_congested_schedule"] == 1.0
    assert third_match["away_congested_schedule"] == 1.0


def test_configurable_congested_threshold() -> None:
    """Configured congestion threshold should control congestion flags."""
    generator = RestDaysGenerator(config={"congested_days": 2})
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-04"],
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Chelsea"],
        }
    )

    result = generator.generate(df)

    assert result.iloc[1]["home_rest_days"] == 3.0
    assert result.iloc[1]["home_congested_schedule"] == 0.0
    assert result.iloc[1]["away_congested_schedule"] == 0.0


def test_long_break_flag(generator: RestDaysGenerator) -> None:
    """A long break for either team should set long_break_flag."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-20"],
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool"],
        }
    )

    result = generator.generate(df)

    assert result.iloc[1]["home_rest_days"] == 19.0
    assert result.iloc[1]["away_rest_days"] == 0.0
    assert result.iloc[1]["long_break_flag"] == 1.0


def test_configurable_long_break_threshold() -> None:
    """Configured long-break threshold should control long-break flags."""
    generator = RestDaysGenerator(config={"long_break_days": 7})
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-08"],
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool"],
        }
    )

    result = generator.generate(df)

    assert result.iloc[1]["home_rest_days"] == 7.0
    assert result.iloc[1]["long_break_flag"] == 1.0


def test_future_leakage_is_prevented(generator: RestDaysGenerator) -> None:
    """Future matches must not affect earlier rest-day features."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-10", "2024-01-01", "2024-01-03"],
            "HomeTeam": ["Arsenal", "Arsenal", "Chelsea"],
            "AwayTeam": ["Chelsea", "Liverpool", "Arsenal"],
        }
    )

    result = generator.generate(df)
    jan_first = result.iloc[1]
    jan_third = result.iloc[2]

    assert jan_first["home_rest_days"] == 0.0
    assert jan_first["away_rest_days"] == 0.0
    assert jan_third["away_rest_days"] == 2.0


def test_same_date_matches_are_not_counted_as_history(
    generator: RestDaysGenerator,
) -> None:
    """Matches on the same date should not be included by the exclusive cutoff."""
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01"],
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool"],
        }
    )

    result = generator.generate(df)

    assert result.iloc[1]["home_rest_days"] == 0.0


def test_generator_uses_team_history_get_last_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RestDaysGenerator should delegate historical lookup to TeamHistory."""
    calls = []
    original = TeamHistory.get_last_matches

    def spy(self: TeamHistory, *args: object, **kwargs: object) -> pd.DataFrame:
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TeamHistory, "get_last_matches", spy)
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-08"],
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Liverpool"],
        }
    )

    RestDaysGenerator().generate(df)

    assert len(calls) == 4
    assert all(call[1]["limit"] == 1 for call in calls)


def test_missing_required_columns_raises_error(generator: RestDaysGenerator) -> None:
    """Generator should reject datasets missing required columns."""
    df = pd.DataFrame({"Date": ["2024-01-01"], "HomeTeam": ["Arsenal"]})

    with pytest.raises(ValueError):
        generator.generate(df)


def test_invalid_threshold_raises_error() -> None:
    """Thresholds must be non-negative."""
    with pytest.raises(ValueError):
        RestDaysGenerator(config={"congested_days": -1})
