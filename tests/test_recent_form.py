"""
Comprehensive tests for the RecentFormGenerator.

These tests verify that recent form features are calculated correctly,
work chronologically, and handle edge cases properly.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import pytest

from backend.app.features.recent_form import RecentFormGenerator
from backend.app.features.registry import FeatureRegistry


@pytest.fixture
def generator() -> RecentFormGenerator:
    """Create a fresh RecentFormGenerator instance for testing."""
    return RecentFormGenerator()


@pytest.fixture
def sample_matches_chronological() -> pd.DataFrame:
    """Create a sample dataset with chronological matches for basic testing."""
    data = {
        "Date": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
            "2024-01-07",
        ],
        "HomeTeam": [
            "Arsenal",
            "Arsenal",
            "Arsenal",
            "Arsenal",
            "Arsenal",
            "Chelsea",
            "Arsenal",
        ],
        "AwayTeam": [
            "Chelsea",
            "Liverpool",
            "ManCity",
            "Spurs",
            "Everton",
            "Arsenal",
            "Fulham",
        ],
        "FTHG": [2, 1, 3, 0, 2, 1, 4],
        "FTAG": [1, 1, 0, 2, 1, 2, 1],
        "FTR": ["H", "D", "H", "A", "H", "H", "H"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def minimal_match() -> pd.DataFrame:
    """Create a minimal single-match dataset."""
    data = {
        "Date": ["2024-01-01"],
        "HomeTeam": ["Arsenal"],
        "AwayTeam": ["Chelsea"],
        "FTHG": [2],
        "FTAG": [1],
        "FTR": ["H"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def matches_with_duplicates() -> pd.DataFrame:
    """Create a dataset with teams having the same matches on the same date."""
    data = {
        "Date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "HomeTeam": ["Arsenal", "Liverpool", "Arsenal", "Liverpool"],
        "AwayTeam": ["Chelsea", "ManCity", "Liverpool", "Arsenal"],
        "FTHG": [2, 1, 3, 2],
        "FTAG": [1, 2, 0, 3],
        "FTR": ["H", "A", "H", "A"],
    }
    return pd.DataFrame(data)


class TestRecentFormGeneratorRegistration:
    """Test that the generator is properly registered."""

    def test_generator_is_registered_in_registry(self) -> None:
        """Recent form generator should be registered automatically."""
        FeatureRegistry.reset()
        # Re-import to trigger registration
        import importlib
        import backend.app.features.recent_form
        importlib.reload(backend.app.features.recent_form)

        registered_names = FeatureRegistry.list_generators()
        assert "recent_form" in registered_names

    def test_generator_can_be_instantiated_from_registry(self) -> None:
        """Generator should be instantiable from the registry."""
        FeatureRegistry.reset()
        import importlib
        import backend.app.features.recent_form
        importlib.reload(backend.app.features.recent_form)

        generator_cls = FeatureRegistry.get("recent_form")
        instance = generator_cls()
        assert instance.name == "recent_form"


class TestRecentFormGeneratorBasics:
    """Test basic functionality of the RecentFormGenerator."""

    def test_generator_has_correct_name(self, generator: RecentFormGenerator) -> None:
        """Generator should have the name 'recent_form'."""
        assert generator.name == "recent_form"

    def test_generator_lists_required_columns(
        self, generator: RecentFormGenerator
    ) -> None:
        """Generator should list all required input columns."""
        required = generator.required_columns
        assert "Date" in required
        assert "HomeTeam" in required
        assert "AwayTeam" in required
        assert "FTHG" in required
        assert "FTAG" in required
        assert "FTR" in required

    def test_generator_lists_output_columns(
        self, generator: RecentFormGenerator
    ) -> None:
        """Generator should list all output feature columns."""
        output_cols = generator.output_columns
        assert "home_form_points_last5" in output_cols
        assert "away_form_points_last5" in output_cols
        assert "home_wins_last5" in output_cols
        assert "away_wins_last5" in output_cols
        assert "home_draws_last5" in output_cols
        assert "away_draws_last5" in output_cols
        assert "home_losses_last5" in output_cols
        assert "away_losses_last5" in output_cols
        assert "home_goals_scored_last5" in output_cols
        assert "away_goals_scored_last5" in output_cols
        assert "home_goals_conceded_last5" in output_cols
        assert "away_goals_conceded_last5" in output_cols
        assert "home_goal_difference_last5" in output_cols
        assert "away_goal_difference_last5" in output_cols
        assert "home_points_per_match" in output_cols
        assert "away_points_per_match" in output_cols

    def test_generator_supports_required_columns(
        self, generator: RecentFormGenerator, minimal_match: pd.DataFrame
    ) -> None:
        """Generator should support datasets with required columns."""
        assert generator.supports(minimal_match) is True

    def test_generator_does_not_support_missing_columns(
        self, generator: RecentFormGenerator
    ) -> None:
        """Generator should not support datasets missing required columns."""
        incomplete_df = pd.DataFrame(
            {"Date": ["2024-01-01"], "HomeTeam": ["Arsenal"]}
        )
        assert generator.supports(incomplete_df) is False


class TestRecentFormGeneratorOutput:
    """Test the output shape and structure of the generator."""

    def test_generate_returns_dataframe(
        self, generator: RecentFormGenerator, minimal_match: pd.DataFrame
    ) -> None:
        """Generate should return a DataFrame."""
        result = generator.generate(minimal_match)
        assert isinstance(result, pd.DataFrame)

    def test_generate_returns_correct_columns(
        self, generator: RecentFormGenerator, minimal_match: pd.DataFrame
    ) -> None:
        """Generate should return exactly the declared output columns."""
        result = generator.generate(minimal_match)
        assert set(result.columns) == set(generator.output_columns)

    def test_generate_preserves_row_count(
        self, generator: RecentFormGenerator, sample_matches_chronological: pd.DataFrame
    ) -> None:
        """Generate should return same number of rows as input."""
        result = generator.generate(sample_matches_chronological)
        assert len(result) == len(sample_matches_chronological)

    def test_generate_returns_numeric_values(
        self, generator: RecentFormGenerator, minimal_match: pd.DataFrame
    ) -> None:
        """Generate should return numeric feature values."""
        result = generator.generate(minimal_match)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])


class TestChronologicalProcessing:
    """Test that the generator processes matches chronologically."""

    def test_first_match_has_zero_features(
        self, generator: RecentFormGenerator, sample_matches_chronological: pd.DataFrame
    ) -> None:
        """First match should have zero form features (no prior matches)."""
        result = generator.generate(sample_matches_chronological)
        first_row = result.iloc[0]

        # First match has no history
        assert first_row["home_form_points_last5"] == 0.0
        assert first_row["away_form_points_last5"] == 0.0
        assert first_row["home_wins_last5"] == 0.0
        assert first_row["away_wins_last5"] == 0.0

    def test_features_accumulate_chronologically(
        self, generator: RecentFormGenerator, sample_matches_chronological: pd.DataFrame
    ) -> None:
        """Form features should accumulate as we move through matches chronologically."""
        result = generator.generate(sample_matches_chronological)

        # Arsenal's first 5 home matches: Win(3), Draw(1), Win(3), Loss(0), Win(3) = 10 points
        arsenal_fourth_match = result.iloc[3]
        assert arsenal_fourth_match["home_form_points_last5"] == 7.0  # First 3 matches: 3+1+3

        # By the 5th match, Arsenal should have 4 prior home matches
        arsenal_fifth_match = result.iloc[4]
        assert arsenal_fifth_match["home_form_points_last5"] == 7.0  # First 4 matches


class TestFormPointsCalculation:
    """Test accurate calculation of form points."""

    def test_home_win_awards_three_points(self, generator: RecentFormGenerator) -> None:
        """Home team should get 3 points for a win."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "HomeTeam": ["Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool"],
                "FTHG": [2, 1],
                "FTAG": [1, 0],
                "FTR": ["H", "H"],
            }
        )
        result = generator.generate(df)

        # Second match: Arsenal has 1 prior win (3 points)
        second_match = result.iloc[1]
        assert second_match["home_form_points_last5"] == 3.0

    def test_draw_awards_one_point(self, generator: RecentFormGenerator) -> None:
        """Team should get 1 point for a draw."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "HomeTeam": ["Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool"],
                "FTHG": [1, 1],
                "FTAG": [1, 1],
                "FTR": ["D", "D"],
            }
        )
        result = generator.generate(df)

        # Second match: Arsenal has 1 prior draw (1 point)
        second_match = result.iloc[1]
        assert second_match["home_form_points_last5"] == 1.0

    def test_loss_awards_zero_points(self, generator: RecentFormGenerator) -> None:
        """Team should get 0 points for a loss."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "HomeTeam": ["Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool"],
                "FTHG": [1, 0],
                "FTAG": [2, 1],
                "FTR": ["A", "A"],
            }
        )
        result = generator.generate(df)

        # Second match: Arsenal has 1 prior loss (0 points)
        second_match = result.iloc[1]
        assert second_match["home_form_points_last5"] == 0.0


class TestAwayTeamFeatures:
    """Test that away team features are calculated correctly."""

    def test_away_team_gets_features(
        self, generator: RecentFormGenerator, sample_matches_chronological: pd.DataFrame
    ) -> None:
        """Away teams should receive form features."""
        result = generator.generate(sample_matches_chronological)

        # Check that away features exist and are numeric
        for idx in range(len(result)):
            assert pd.notna(result.loc[idx, "away_form_points_last5"])
            assert isinstance(result.loc[idx, "away_form_points_last5"], (int, float))

    def test_away_win_in_draw_is_loss(self, generator: RecentFormGenerator) -> None:
        """Away team with result 'D' should not get win."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "HomeTeam": ["Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool"],
                "FTHG": [1, 1],
                "FTAG": [1, 1],
                "FTR": ["D", "D"],
            }
        )
        result = generator.generate(df)

        # Chelsea: first match is a draw (as away team), so 1 point, 0 wins
        # Liverpool: has no prior history, so 0 points
        second_match = result.iloc[1]
        assert second_match["away_form_points_last5"] == 0.0
        assert second_match["away_wins_last5"] == 0.0


class TestLast5MatchesWindow:
    """Test that the rolling 5-match window works correctly."""

    def test_only_last_five_matches_counted(
        self, generator: RecentFormGenerator
    ) -> None:
        """Only the last 5 matches should be used in calculations."""
        # Create 8 matches for one team
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=8),
                "HomeTeam": ["Arsenal"] * 8,
                "AwayTeam": ["Chelsea", "Liverpool", "ManCity", "Spurs", "Everton", "Fulham", "Brentford", "Brighton"],
                "FTHG": [3, 3, 3, 3, 3, 3, 3, 3],  # All wins
                "FTAG": [0, 0, 0, 0, 0, 0, 0, 0],
                "FTR": ["H", "H", "H", "H", "H", "H", "H", "H"],
            }
        )
        result = generator.generate(df)

        # By match 6, Arsenal has had 5+ matches, so last5 should be capped at 15 points
        match_6 = result.iloc[5]
        assert match_6["home_form_points_last5"] == 15.0  # 5 wins * 3 points

        # Match 7 still has only 5 in window (matches 2-6)
        match_7 = result.iloc[6]
        assert match_7["home_form_points_last5"] == 15.0

        # Match 8 has matches 3-7 (match 1 falls out of window)
        match_8 = result.iloc[7]
        assert match_8["home_form_points_last5"] == 15.0

    def test_fewer_than_five_matches_uses_available(
        self, generator: RecentFormGenerator
    ) -> None:
        """Teams with fewer than 5 prior matches should use available matches."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "HomeTeam": ["Arsenal", "Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool", "ManCity"],
                "FTHG": [3, 3, 3],
                "FTAG": [0, 0, 0],
                "FTR": ["H", "H", "H"],
            }
        )
        result = generator.generate(df)

        # Match 2: 1 prior match (3 points)
        match_2 = result.iloc[1]
        assert match_2["home_form_points_last5"] == 3.0

        # Match 3: 2 prior matches (6 points)
        match_3 = result.iloc[2]
        assert match_3["home_form_points_last5"] == 6.0


class TestGoalStatistics:
    """Test calculation of goal-related features."""

    def test_goals_scored_accumulated(self, generator: RecentFormGenerator) -> None:
        """Goals scored should accumulate correctly."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "HomeTeam": ["Arsenal", "Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool", "ManCity"],
                "FTHG": [2, 1, 3],
                "FTAG": [0, 0, 0],
                "FTR": ["H", "H", "H"],
            }
        )
        result = generator.generate(df)

        # Match 2: Arsenal scored 2 in match 1
        match_2 = result.iloc[1]
        assert match_2["home_goals_scored_last5"] == 2.0

        # Match 3: Arsenal scored 2+1=3 in matches 1-2
        match_3 = result.iloc[2]
        assert match_3["home_goals_scored_last5"] == 3.0

    def test_goals_conceded_accumulated(self, generator: RecentFormGenerator) -> None:
        """Goals conceded should accumulate correctly."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "HomeTeam": ["Arsenal", "Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool", "ManCity"],
                "FTHG": [2, 1, 3],
                "FTAG": [1, 2, 1],
                "FTR": ["H", "A", "H"],
            }
        )
        result = generator.generate(df)

        # Match 2: Arsenal conceded 1 in match 1
        match_2 = result.iloc[1]
        assert match_2["home_goals_conceded_last5"] == 1.0

        # Match 3: Arsenal conceded 1+2=3 in matches 1-2
        match_3 = result.iloc[2]
        assert match_3["home_goals_conceded_last5"] == 3.0

    def test_goal_difference_calculated_correctly(
        self, generator: RecentFormGenerator
    ) -> None:
        """Goal difference should be goals scored minus goals conceded."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "HomeTeam": ["Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool"],
                "FTHG": [3, 2],
                "FTAG": [1, 2],
                "FTR": ["H", "D"],
            }
        )
        result = generator.generate(df)

        # Match 2: Arsenal scored 3, conceded 1, so +2
        match_2 = result.iloc[1]
        assert match_2["home_goals_scored_last5"] == 3.0
        assert match_2["home_goals_conceded_last5"] == 1.0
        assert match_2["home_goal_difference_last5"] == 2.0

    def test_negative_goal_difference(
        self, generator: RecentFormGenerator
    ) -> None:
        """Goal difference can be negative."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "HomeTeam": ["Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool"],
                "FTHG": [1, 0],
                "FTAG": [3, 2],
                "FTR": ["A", "A"],
            }
        )
        result = generator.generate(df)

        # Match 2: Arsenal scored 1, conceded 3, so -2
        match_2 = result.iloc[1]
        assert match_2["home_goals_scored_last5"] == 1.0
        assert match_2["home_goals_conceded_last5"] == 3.0
        assert match_2["home_goal_difference_last5"] == -2.0


class TestWinsDrawsLosses:
    """Test calculation of match result counts."""

    def test_wins_counted_correctly(self, generator: RecentFormGenerator) -> None:
        """Wins should be counted accurately."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "HomeTeam": ["Arsenal", "Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool", "ManCity"],
                "FTHG": [2, 1, 0],
                "FTAG": [1, 0, 1],
                "FTR": ["H", "H", "A"],
            }
        )
        result = generator.generate(df)

        # Match 3: Arsenal has 2 prior wins (matches 1-2)
        # When calculating for match 3, we use history up to match 2
        match_3 = result.iloc[2]
        # Arsenal's history for match 3: 2 wins (matches 1-2) = 6 points
        assert match_3["home_wins_last5"] == 2.0
        assert match_3["home_losses_last5"] == 0.0

    def test_draws_counted_correctly(self, generator: RecentFormGenerator) -> None:
        """Draws should be counted accurately."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "HomeTeam": ["Arsenal", "Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool", "ManCity"],
                "FTHG": [1, 1, 1],
                "FTAG": [1, 1, 0],
                "FTR": ["D", "D", "H"],
            }
        )
        result = generator.generate(df)

        # Match 3: Arsenal has prior matches (draws in 1-2)
        # When calculating for match 3, we use history up to match 2
        match_3 = result.iloc[2]
        # Arsenal's history for match 3: 2 draws (matches 1-2) = 2 points, 0 wins
        assert match_3["home_draws_last5"] == 2.0
        assert match_3["home_wins_last5"] == 0.0

    def test_losses_counted_correctly(self, generator: RecentFormGenerator) -> None:
        """Losses should be counted accurately."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "HomeTeam": ["Arsenal", "Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool", "ManCity"],
                "FTHG": [0, 1, 0],
                "FTAG": [1, 2, 2],
                "FTR": ["A", "A", "A"],
            }
        )
        result = generator.generate(df)

        # Match 3: Arsenal has 3 losses
        match_3 = result.iloc[2]
        assert match_3["home_losses_last5"] == 2.0


class TestPointsPerMatch:
    """Test calculation of points per match metric."""

    def test_points_per_match_with_all_wins(
        self, generator: RecentFormGenerator
    ) -> None:
        """Points per match should be 3.0 when all matches are wins."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "HomeTeam": ["Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool"],
                "FTHG": [3, 2],
                "FTAG": [0, 0],
                "FTR": ["H", "H"],
            }
        )
        result = generator.generate(df)

        # Match 2: 1 prior win (3 points / 1 match)
        match_2 = result.iloc[1]
        assert match_2["home_points_per_match"] == 3.0

    def test_points_per_match_with_mixed_results(
        self, generator: RecentFormGenerator
    ) -> None:
        """Points per match should average correctly."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "HomeTeam": ["Arsenal", "Arsenal", "Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool", "ManCity", "Spurs"],
                "FTHG": [3, 1, 1, 0],
                "FTAG": [0, 1, 0, 1],
                "FTR": ["H", "D", "H", "A"],
            }
        )
        result = generator.generate(df)

        # Match 4: 3 prior results (Win=3, Draw=1, Win=3) = 7 points / 3 matches = 2.33...
        match_4 = result.iloc[3]
        expected_ppm = 7.0 / 3.0
        assert match_4["home_points_per_match"] == pytest.approx(expected_ppm, rel=1e-5)

    def test_points_per_match_with_no_prior_matches(
        self, generator: RecentFormGenerator
    ) -> None:
        """Points per match should be 0.0 when no prior matches exist."""
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

        # First match: no prior history
        match_1 = result.iloc[0]
        assert match_1["home_points_per_match"] == 0.0


class TestSeparateTeamHistories:
    """Test that each team maintains its own history."""

    def test_home_and_away_records_combined(
        self, generator: RecentFormGenerator
    ) -> None:
        """Home and away results are combined in team history for form calculation."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
                "AwayTeam": ["Chelsea", "Arsenal", "Liverpool"],
                "FTHG": [2, 1, 3],
                "FTAG": [1, 0, 0],
                "FTR": ["H", "H", "H"],
            }
        )
        result = generator.generate(df)

        # Match 2: Arsenal played as home team (Win=3 pts) and now plays as away team
        match_2 = result.iloc[1]
        # Arsenal (away) has 1 prior home win = 3 points in combined history
        assert match_2["away_form_points_last5"] == 3.0

        # Match 3: Arsenal (home) has 2 prior matches in history (home win + away loss)
        match_3 = result.iloc[2]
        # Arsenal's total record: home win (3) + away loss (0) = 3 points
        assert match_3["home_form_points_last5"] == 3.0

    def test_different_teams_have_independent_histories(
        self, generator: RecentFormGenerator
    ) -> None:
        """Different teams should not share history."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "HomeTeam": ["Arsenal", "Arsenal", "Chelsea", "Chelsea"],
                "AwayTeam": ["Chelsea", "Liverpool", "Arsenal", "Liverpool"],
                "FTHG": [2, 1, 1, 2],
                "FTAG": [1, 0, 2, 1],
                "FTR": ["H", "H", "A", "H"],
            }
        )
        result = generator.generate(df)

        # Match 4: Chelsea (home) should have their own history
        # Chelsea has: Away loss (match 1, 0 pts), Home loss (match 3, 0 pts)
        match_4 = result.iloc[3]
        # Chelsea's record: loss (0) + loss (0) = 0 points
        assert match_4["home_form_points_last5"] == 0.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe_input(self, generator: RecentFormGenerator) -> None:
        """Generator should handle empty DataFrames gracefully."""
        df = pd.DataFrame(
            {
                "Date": [],
                "HomeTeam": [],
                "AwayTeam": [],
                "FTHG": [],
                "FTAG": [],
                "FTR": [],
            }
        )
        result = generator.generate(df)

        assert len(result) == 0
        assert set(result.columns) == set(generator.output_columns)

    def test_missing_required_columns_raises_error(
        self, generator: RecentFormGenerator
    ) -> None:
        """Generator should raise error when required columns are missing."""
        df = pd.DataFrame(
            {"Date": ["2024-01-01"], "HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"]}
        )
        with pytest.raises(ValueError):
            generator.generate(df)

    def test_zero_goals_handled_correctly(self, generator: RecentFormGenerator) -> None:
        """Goalless matches should be handled correctly."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "HomeTeam": ["Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool"],
                "FTHG": [0, 0],
                "FTAG": [0, 0],
                "FTR": ["D", "D"],
            }
        )
        result = generator.generate(df)

        # Match 2: Arsenal has 1 prior draw with 0 goals
        match_2 = result.iloc[1]
        assert match_2["home_goals_scored_last5"] == 0.0
        assert match_2["home_goals_conceded_last5"] == 0.0
        assert match_2["home_goal_difference_last5"] == 0.0
        assert match_2["home_draws_last5"] == 1.0
        assert match_2["home_form_points_last5"] == 1.0

    def test_high_scoring_match_handled(self, generator: RecentFormGenerator) -> None:
        """High-scoring matches should be handled correctly."""
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "HomeTeam": ["Arsenal", "Arsenal"],
                "AwayTeam": ["Chelsea", "Liverpool"],
                "FTHG": [6, 5],
                "FTAG": [5, 4],
                "FTR": ["H", "H"],
            }
        )
        result = generator.generate(df)

        # Match 2: Arsenal scored 6, conceded 5
        match_2 = result.iloc[1]
        assert match_2["home_goals_scored_last5"] == 6.0
        assert match_2["home_goals_conceded_last5"] == 5.0
        assert match_2["home_goal_difference_last5"] == 1.0


class TestDataTypeConsistency:
    """Test that output data types are consistent and correct."""

    def test_all_features_are_float(
        self, generator: RecentFormGenerator, sample_matches_chronological: pd.DataFrame
    ) -> None:
        """All output features should be float type."""
        result = generator.generate(sample_matches_chronological)

        for col in result.columns:
            assert pd.api.types.is_float_dtype(result[col]), f"Column {col} is not float"

    def test_no_nan_in_output(
        self, generator: RecentFormGenerator, sample_matches_chronological: pd.DataFrame
    ) -> None:
        """No NaN values should appear in output."""
        result = generator.generate(sample_matches_chronological)

        assert result.isna().sum().sum() == 0


class TestIntegrationWithRealWorldScenarios:
    """Test realistic scenarios with multiple teams and complex histories."""

    def test_multi_team_season_simulation(
        self, generator: RecentFormGenerator
    ) -> None:
        """Test a realistic season with multiple teams playing multiple matches."""
        dates = pd.date_range("2024-01-01", periods=10)
        matches = [
            ("2024-01-01", "Arsenal", "Chelsea", 2, 1, "H"),
            ("2024-01-02", "Liverpool", "ManCity", 1, 0, "H"),
            ("2024-01-03", "Arsenal", "Liverpool", 1, 1, "D"),
            ("2024-01-04", "Chelsea", "ManCity", 0, 2, "A"),
            ("2024-01-05", "Arsenal", "ManCity", 3, 1, "H"),
            ("2024-01-06", "Liverpool", "Chelsea", 2, 0, "H"),
            ("2024-01-07", "ManCity", "Arsenal", 0, 1, "A"),
            ("2024-01-08", "Chelsea", "Liverpool", 1, 1, "D"),
            ("2024-01-09", "Arsenal", "Liverpool", 2, 0, "H"),
            ("2024-01-10", "ManCity", "Chelsea", 3, 0, "H"),
        ]

        df = pd.DataFrame(
            matches,
            columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"],
        )

        result = generator.generate(df)

        assert len(result) == len(df)
        assert set(result.columns) == set(generator.output_columns)
        assert result.isna().sum().sum() == 0
