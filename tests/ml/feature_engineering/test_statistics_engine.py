"""
Tests for the StatisticsEngine and calculator functions.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from backend.ml.feature_engineering.exceptions import (
    InvalidWindowError,
    MissingColumnError,
)
from backend.ml.feature_engineering.statistics_engine import StatisticsEngine
from backend.ml.feature_engineering.calculators import (
    calculate_points,
    calculate_is_win,
    calculate_clean_sheet,
)
from backend.ml.feature_engineering.validators import validate_required_columns


@pytest.fixture
def sample_match_data():
    """Provides a sample match history DataFrame."""
    base_date = datetime(2023, 1, 1)
    
    # 6 matches for TeamA (3 home, 3 away)
    # TeamA results: W(3-0), D(1-1), L(0-1), W(2-0), W(1-0), D(0-0)
    # Points: 3, 1, 0, 3, 3, 1 -> total=11 over 6 games
    # Rolling 5 window before the 6th game should contain the first 5 games.
    
    data = [
        {"Date": base_date + timedelta(days=1), "HomeTeam": "TeamA", "AwayTeam": "TeamB", "FTHG": 3, "FTAG": 0, "FTR": "H"},
        {"Date": base_date + timedelta(days=2), "HomeTeam": "TeamC", "AwayTeam": "TeamA", "FTHG": 1, "FTAG": 1, "FTR": "D"},
        {"Date": base_date + timedelta(days=3), "HomeTeam": "TeamA", "AwayTeam": "TeamD", "FTHG": 0, "FTAG": 1, "FTR": "A"},
        {"Date": base_date + timedelta(days=4), "HomeTeam": "TeamE", "AwayTeam": "TeamA", "FTHG": 0, "FTAG": 2, "FTR": "A"},
        {"Date": base_date + timedelta(days=5), "HomeTeam": "TeamA", "AwayTeam": "TeamF", "FTHG": 1, "FTAG": 0, "FTR": "H"},
        {"Date": base_date + timedelta(days=6), "HomeTeam": "TeamG", "AwayTeam": "TeamA", "FTHG": 0, "FTAG": 0, "FTR": "D"},
    ]
    
    return pd.DataFrame(data)


class TestCalculators:
    """Tests for the stateless calculator functions."""
    
    def test_calculate_points(self):
        assert calculate_points('H', True) == 3
        assert calculate_points('A', False) == 3
        assert calculate_points('D', True) == 1
        assert calculate_points('D', False) == 1
        assert calculate_points('A', True) == 0
        assert calculate_points('H', False) == 0

    def test_calculate_is_win(self):
        assert calculate_is_win('H', True) == 1
        assert calculate_is_win('A', False) == 1
        assert calculate_is_win('D', True) == 0
        assert calculate_is_win('A', True) == 0

    def test_calculate_clean_sheet(self):
        assert calculate_clean_sheet(0) == 1
        assert calculate_clean_sheet(1) == 0
        assert calculate_clean_sheet(3) == 0


class TestValidators:
    """Tests for feature engineering validators."""
    
    def test_validate_missing_columns(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        with pytest.raises(MissingColumnError, match="Dataset is missing required columns"):
            validate_required_columns(df, ["A", "C"])
            
    def test_validate_success(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        # Should not raise
        validate_required_columns(df, ["A", "B"])


class TestStatisticsEngine:
    """Tests for the main StatisticsEngine class."""
    
    def test_invalid_window_size(self):
        with pytest.raises(InvalidWindowError):
            StatisticsEngine(window_size=0)
        with pytest.raises(InvalidWindowError):
            StatisticsEngine(window_size=-5)
            
    def test_missing_required_columns(self):
        engine = StatisticsEngine()
        df = pd.DataFrame({"Date": ["2023-01-01"]})
        with pytest.raises(MissingColumnError):
            engine.generate_features(df)
            
    def test_generate_features_adds_columns(self, sample_match_data):
        engine = StatisticsEngine(window_size=5)
        result_df, metadata = engine.generate_features(sample_match_data)
        
        expected_columns = [
            "Home Team Form", "Away Team Form",
            "Home Goals Average", "Away Goals Average",
            "Home Goals Conceded Average", "Away Goals Conceded Average",
            "Home Win Percentage", "Away Win Percentage",
            "Home Clean Sheet Percentage", "Away Clean Sheet Percentage",
            "Home Goal Difference", "Away Goal Difference"
        ]
        
        for col in expected_columns:
            assert col in result_df.columns
            assert col in metadata.features_generated
            
        assert metadata.window_size == 5
        assert metadata.initial_feature_count == 6
        assert metadata.final_feature_count == 18
        
    def test_rolling_calculations_accuracy(self, sample_match_data):
        engine = StatisticsEngine(window_size=3)
        result_df, metadata = engine.generate_features(sample_match_data)
        
        # Look at the 4th match for TeamA (index 3)
        # Prior 3 matches: W(3-0), D(1-1), L(0-1)
        # Points: 3 + 1 + 0 = 4
        # Goals Scored: 3 + 1 + 0 = 4 (avg = 1.33)
        # Goals Conceded: 0 + 1 + 1 = 2 (avg = 0.66)
        # Wins: 1 (win % = 0.33)
        # Clean Sheets: 1 (clean sheet % = 0.33)
        # Goal Diff: (3-0) + (1-1) + (0-1) = 3 + 0 - 1 = 2 (avg = 0.66)
        
        # In match 4, TeamA is AwayTeam ("TeamA")
        match_4 = result_df.iloc[3]
        assert match_4["AwayTeam"] == "TeamA"
        
        assert match_4["Away Team Form"] == 4.0
        assert abs(match_4["Away Goals Average"] - (4/3)) < 1e-5
        assert abs(match_4["Away Goals Conceded Average"] - (2/3)) < 1e-5
        assert abs(match_4["Away Win Percentage"] - (1/3)) < 1e-5
        assert abs(match_4["Away Clean Sheet Percentage"] - (1/3)) < 1e-5
        assert abs(match_4["Away Goal Difference"] - (2/3)) < 1e-5
        
    def test_first_match_is_zero(self, sample_match_data):
        # The first match for any team should have 0s for all historical stats
        engine = StatisticsEngine(window_size=5)
        result_df, _ = engine.generate_features(sample_match_data)
        
        first_match = result_df.iloc[0]
        # TeamA is Home, TeamB is Away (their first match ever)
        assert first_match["Home Team Form"] == 0.0
        assert first_match["Away Team Form"] == 0.0
        assert first_match["Home Goals Average"] == 0.0
        assert first_match["Away Goals Average"] == 0.0
