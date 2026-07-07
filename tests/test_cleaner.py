"""
Unit tests for cleaner module.

Tests cover duplicate removal, team name standardization, date conversion,
data type conversion, missing value handling, and chronological sorting.
"""

import pytest
import pandas as pd
from datetime import datetime

from backend.app.data.cleaner import DataCleaner
from backend.app.data.exceptions import DataProcessingError


@pytest.fixture
def sample_raw_dataframe():
    """Create a sample raw dataframe with typical dirty data."""
    return pd.DataFrame({
        "Date": ["2023-01-03", "2023-01-01", "2023-01-02"],
        "HomeTeam": ["  Arsenal  ", "manchester united", "Liverpool"],
        "AwayTeam": ["Chelsea ", " Manchester City", "Tottenham"],
        "FTHG": ["2", "1", "3"],
        "FTAG": ["1", "1", "0"],
        "FTR": ["H", "D", "H"],
        "HS": ["10", "8", "12"],
        "AS": ["6", "7", "4"],
        "HST": ["5", "3", "7"],
        "AST": ["2", "2", "1"],
        "HC": ["8", "6", "9"],
        "AC": ["5", "7", "3"],
        "HY": ["2", "1", "3"],
        "AY": ["1", "2", "2"],
        "HR": ["0", "0", "1"],
        "AR": ["0", "1", "0"],
    })


@pytest.fixture
def cleaner():
    """Create a DataCleaner instance."""
    return DataCleaner()


class TestRemoveDuplicates:
    """Tests for duplicate removal."""

    def test_remove_no_duplicates(self, cleaner, sample_raw_dataframe):
        """Test cleaning data without duplicates."""
        df = sample_raw_dataframe.copy()
        result = cleaner.remove_duplicates(df)
        assert len(result) == 3

    def test_remove_exact_duplicates(self, cleaner, sample_raw_dataframe):
        """Test removal of exact duplicate rows."""
        df = pd.concat([sample_raw_dataframe, sample_raw_dataframe.iloc[[0]]])
        result = cleaner.remove_duplicates(df)
        assert len(result) == 3

    def test_remove_duplicates_keeps_first(self, cleaner, sample_raw_dataframe):
        """Test that first occurrence is kept."""
        df = sample_raw_dataframe.copy()
        first_row = df.iloc[0].copy()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        result = cleaner.remove_duplicates(df)

        assert len(result) == 3
        assert result.iloc[0].equals(first_row)

    def test_remove_multiple_duplicates(self, cleaner, sample_raw_dataframe):
        """Test removal of multiple duplicate sets."""
        df = pd.concat([
            sample_raw_dataframe,
            sample_raw_dataframe.iloc[[0]],
            sample_raw_dataframe.iloc[[1]],
        ], ignore_index=True)

        result = cleaner.remove_duplicates(df)
        assert len(result) == 3

    def test_remove_duplicates_preserves_columns(self, cleaner, sample_raw_dataframe):
        """Test that columns are preserved."""
        df = pd.concat([sample_raw_dataframe, sample_raw_dataframe.iloc[[0]]])
        result = cleaner.remove_duplicates(df)
        assert list(result.columns) == list(sample_raw_dataframe.columns)


class TestStandardizeTeamNames:
    """Tests for team name standardization."""

    def test_standardize_remove_leading_spaces(self, cleaner):
        """Test removal of leading spaces."""
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "HomeTeam": ["  Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": ["2"],
            "FTAG": ["1"],
            "FTR": ["H"],
            "HS": ["10"],
            "AS": ["6"],
            "HST": ["5"],
            "AST": ["2"],
            "HC": ["8"],
            "AC": ["5"],
            "HY": ["2"],
            "AY": ["1"],
            "HR": ["0"],
            "AR": ["0"],
        })

        result = cleaner.standardize_team_names(df)
        assert result.loc[0, "HomeTeam"] == "Arsenal"

    def test_standardize_remove_trailing_spaces(self, cleaner):
        """Test removal of trailing spaces."""
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea  "],
            "FTHG": ["2"],
            "FTAG": ["1"],
            "FTR": ["H"],
            "HS": ["10"],
            "AS": ["6"],
            "HST": ["5"],
            "AST": ["2"],
            "HC": ["8"],
            "AC": ["5"],
            "HY": ["2"],
            "AY": ["1"],
            "HR": ["0"],
            "AR": ["0"],
        })

        result = cleaner.standardize_team_names(df)
        assert result.loc[0, "AwayTeam"] == "Chelsea"

    def test_standardize_all_team_columns(self, cleaner):
        """Test standardization applied to all team columns."""
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "HomeTeam": ["  Arsenal  "],
            "AwayTeam": ["  Chelsea  "],
            "FTHG": ["2"],
            "FTAG": ["1"],
            "FTR": ["H"],
            "HS": ["10"],
            "AS": ["6"],
            "HST": ["5"],
            "AST": ["2"],
            "HC": ["8"],
            "AC": ["5"],
            "HY": ["2"],
            "AY": ["1"],
            "HR": ["0"],
            "AR": ["0"],
        })

        result = cleaner.standardize_team_names(df)
        assert result.loc[0, "HomeTeam"] == "Arsenal"
        assert result.loc[0, "AwayTeam"] == "Chelsea"

    def test_standardize_without_team_columns(self, cleaner):
        """Test standardization when team columns missing."""
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "FTHG": ["2"],
        })

        result = cleaner.standardize_team_names(df)
        assert len(result) == 1


class TestConvertDates:
    """Tests for date conversion."""

    def test_convert_valid_dates(self, cleaner):
        """Test conversion of valid date strings."""
        df = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Liverpool", "Manchester"],
            "FTHG": ["2", "1"],
            "FTAG": ["1", "0"],
            "FTR": ["H", "H"],
            "HS": ["10", "8"],
            "AS": ["6", "4"],
            "HST": ["5", "3"],
            "AST": ["2", "1"],
            "HC": ["8", "6"],
            "AC": ["5", "3"],
            "HY": ["2", "1"],
            "AY": ["1", "0"],
            "HR": ["0", "0"],
            "AR": ["0", "0"],
        })

        result = cleaner.convert_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_convert_dates_without_date_column(self, cleaner):
        """Test conversion when Date column missing."""
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal"],
            "FTHG": ["2"],
        })

        result = cleaner.convert_dates(df)
        assert len(result) == 1

    def test_convert_mixed_date_formats(self, cleaner):
        """Test conversion of mixed date formats."""
        df = pd.DataFrame({
            "Date": ["2023-01-01", "01/01/2023"],
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Liverpool", "Manchester"],
            "FTHG": ["2", "1"],
            "FTAG": ["1", "0"],
            "FTR": ["H", "H"],
            "HS": ["10", "8"],
            "AS": ["6", "4"],
            "HST": ["5", "3"],
            "AST": ["2", "1"],
            "HC": ["8", "6"],
            "AC": ["5", "3"],
            "HY": ["2", "1"],
            "AY": ["1", "0"],
            "HR": ["0", "0"],
            "AR": ["0", "0"],
        })

        result = cleaner.convert_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])


class TestConvertDataTypes:
    """Tests for data type conversion."""

    def test_convert_numeric_columns_to_int(self, cleaner, sample_raw_dataframe):
        """Test conversion of numeric columns to int64."""
        df = sample_raw_dataframe.copy()
        result = cleaner.convert_data_types(df)

        numeric_cols = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]
        for col in numeric_cols:
            assert result[col].dtype == "int64"

    def test_convert_categorical_to_string(self, cleaner, sample_raw_dataframe):
        """Test conversion of categorical columns to string."""
        df = sample_raw_dataframe.copy()
        result = cleaner.convert_data_types(df)

        assert result["HomeTeam"].dtype == "object"
        assert result["AwayTeam"].dtype == "object"

    def test_convert_date_to_datetime(self, cleaner, sample_raw_dataframe):
        """Test conversion of date to datetime64."""
        df = sample_raw_dataframe.copy()
        result = cleaner.convert_data_types(df)

        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_convert_missing_columns(self, cleaner):
        """Test conversion with missing columns."""
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "HomeTeam": ["Arsenal"],
        })

        result = cleaner.convert_data_types(df)
        assert len(result) == 1


class TestFillMissingValues:
    """Tests for filling missing values."""

    def test_fill_numeric_nulls_with_zero(self, cleaner):
        """Test filling numeric nulls with 0."""
        df = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Liverpool", "Manchester"],
            "FTHG": ["2", None],
            "FTAG": ["1", "0"],
            "FTR": ["H", "H"],
            "HS": ["10", None],
            "AS": ["6", "4"],
            "HST": ["5", "3"],
            "AST": ["2", "1"],
            "HC": ["8", "6"],
            "AC": ["5", None],
            "HY": ["2", "1"],
            "AY": ["1", "0"],
            "HR": ["0", "0"],
            "AR": ["0", "0"],
        })

        result = cleaner.fill_missing_values(df)
        assert result.loc[1, "FTHG"] == 0

    def test_fill_categorical_nulls(self, cleaner):
        """Test filling categorical nulls."""
        df = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Liverpool", None],
            "FTHG": ["2", "1"],
            "FTAG": ["1", "0"],
            "FTR": ["H", "H"],
            "HS": ["10", "8"],
            "AS": ["6", "4"],
            "HST": ["5", "3"],
            "AST": ["2", "1"],
            "HC": ["8", "6"],
            "AC": ["5", "3"],
            "HY": ["2", "1"],
            "AY": ["1", "0"],
            "HR": ["0", "0"],
            "AR": ["0", "0"],
        })

        result = cleaner.fill_missing_values(df)
        assert result.loc[1, "AwayTeam"] == "Unknown"

    def test_remove_rows_with_critical_nulls(self, cleaner):
        """Test removal of rows with critical missing values."""
        df = pd.DataFrame({
            "Date": ["2023-01-01", None, "2023-01-03"],
            "HomeTeam": ["Arsenal", "Chelsea", "Liverpool"],
            "AwayTeam": ["Manchester", "Manchester", None],
            "FTHG": ["2", "1", "3"],
            "FTAG": ["1", "0", "0"],
            "FTR": ["H", "H", "H"],
            "HS": ["10", "8", "12"],
            "AS": ["6", "4", "5"],
            "HST": ["5", "3", "7"],
            "AST": ["2", "1", "2"],
            "HC": ["8", "6", "9"],
            "AC": ["5", "3", "4"],
            "HY": ["2", "1", "3"],
            "AY": ["1", "0", "1"],
            "HR": ["0", "0", "1"],
            "AR": ["0", "0", "0"],
        })

        result = cleaner.fill_missing_values(df)
        assert len(result) == 1


class TestSortChronologically:
    """Tests for chronological sorting."""

    def test_sort_unsorted_dates(self, cleaner, sample_raw_dataframe):
        """Test sorting of unsorted dates."""
        result = cleaner.sort_chronologically(sample_raw_dataframe)

        dates = pd.to_datetime(result["Date"])
        assert list(dates) == sorted(dates.tolist())

    def test_sort_already_sorted(self, cleaner):
        """Test sorting already sorted data."""
        df = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "HomeTeam": ["Arsenal", "Chelsea", "Liverpool"],
            "AwayTeam": ["Manchester", "Manchester", "Tottenham"],
            "FTHG": ["2", "1", "3"],
            "FTAG": ["1", "0", "0"],
            "FTR": ["H", "H", "H"],
            "HS": ["10", "8", "12"],
            "AS": ["6", "4", "5"],
            "HST": ["5", "3", "7"],
            "AST": ["2", "1", "2"],
            "HC": ["8", "6", "9"],
            "AC": ["5", "3", "4"],
            "HY": ["2", "1", "3"],
            "AY": ["1", "0", "1"],
            "HR": ["0", "0", "1"],
            "AR": ["0", "0", "0"],
        })

        result = cleaner.sort_chronologically(df)
        assert len(result) == 3

    def test_sort_without_date_column(self, cleaner):
        """Test sorting when Date column missing."""
        df = pd.DataFrame({
            "HomeTeam": ["Arsenal", "Chelsea"],
            "FTHG": ["2", "1"],
        })

        result = cleaner.sort_chronologically(df)
        assert len(result) == 2


class TestCompleteCleaning:
    """Tests for complete cleaning pipeline."""

    def test_complete_cleaning_pipeline(self, cleaner, sample_raw_dataframe):
        """Test complete cleaning process."""
        result = cleaner.clean(sample_raw_dataframe)

        assert len(result) > 0
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])
        assert all(result.index == range(len(result)))

    def test_cleaning_removes_duplicates(self, cleaner, sample_raw_dataframe):
        """Test that cleaning removes duplicates."""
        df = pd.concat([sample_raw_dataframe, sample_raw_dataframe.iloc[[0]]])
        result = cleaner.clean(df)

        assert len(result) == 3

    def test_cleaning_standardizes_names(self, cleaner):
        """Test that cleaning standardizes team names."""
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "HomeTeam": ["  Arsenal  "],
            "AwayTeam": ["  Chelsea  "],
            "FTHG": ["2"],
            "FTAG": ["1"],
            "FTR": ["H"],
            "HS": ["10"],
            "AS": ["6"],
            "HST": ["5"],
            "AST": ["2"],
            "HC": ["8"],
            "AC": ["5"],
            "HY": ["2"],
            "AY": ["1"],
            "HR": ["0"],
            "AR": ["0"],
        })

        result = cleaner.clean(df)
        assert result.loc[0, "HomeTeam"] == "Arsenal"
        assert result.loc[0, "AwayTeam"] == "Chelsea"

    def test_cleaning_sorts_dates(self, cleaner, sample_raw_dataframe):
        """Test that cleaning sorts by date."""
        result = cleaner.clean(sample_raw_dataframe)

        dates = pd.to_datetime(result["Date"])
        assert list(dates) == sorted(dates.tolist())


class TestTeamNameMapping:
    """Tests for team name mapping."""

    def test_register_team_mapping(self, cleaner):
        """Test registering team name mappings."""
        mapping = {"Arsenal FC": "Arsenal", "Manchester Utd": "Manchester United"}
        cleaner.register_team_name_mapping(mapping)

        assert cleaner._team_name_mapping == mapping

    def test_apply_team_mapping(self, cleaner):
        """Test applying team name mappings."""
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "HomeTeam": ["Arsenal FC"],
            "AwayTeam": ["Manchester Utd"],
            "FTHG": ["2"],
            "FTAG": ["1"],
            "FTR": ["H"],
            "HS": ["10"],
            "AS": ["6"],
            "HST": ["5"],
            "AST": ["2"],
            "HC": ["8"],
            "AC": ["5"],
            "HY": ["2"],
            "AY": ["1"],
            "HR": ["0"],
            "AR": ["0"],
        })

        mapping = {"Arsenal FC": "Arsenal", "Manchester Utd": "Manchester United"}
        cleaner.register_team_name_mapping(mapping)
        result = cleaner.apply_team_name_mapping(df)

        assert result.loc[0, "HomeTeam"] == "Arsenal"
        assert result.loc[0, "AwayTeam"] == "Manchester United"

    def test_apply_mapping_unmapped_teams(self, cleaner):
        """Test that unmapped teams are unchanged."""
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "FTHG": ["2"],
            "FTAG": ["1"],
            "FTR": ["H"],
            "HS": ["10"],
            "AS": ["6"],
            "HST": ["5"],
            "AST": ["2"],
            "HC": ["8"],
            "AC": ["5"],
            "HY": ["2"],
            "AY": ["1"],
            "HR": ["0"],
            "AR": ["0"],
        })

        mapping = {"Arsenal FC": "Arsenal"}
        cleaner.register_team_name_mapping(mapping)
        result = cleaner.apply_team_name_mapping(df)

        assert result.loc[0, "HomeTeam"] == "Arsenal"
        assert result.loc[0, "AwayTeam"] == "Liverpool"


class TestErrorHandling:
    """Tests for error handling."""

    def test_error_on_invalid_operation(self, cleaner):
        """Test that invalid operations raise errors."""
        df = None

        with pytest.raises(DataProcessingError):
            cleaner.remove_duplicates(df)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_clean_empty_dataframe(self, cleaner):
        """Test cleaning empty dataframe."""
        df = pd.DataFrame(columns=[
            "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"
        ])

        result = cleaner.clean(df)
        assert len(result) == 0

    def test_clean_single_row(self, cleaner, sample_raw_dataframe):
        """Test cleaning single row."""
        df = sample_raw_dataframe.iloc[[0]]
        result = cleaner.clean(df)
        assert len(result) == 1

    def test_clean_large_dataset(self, cleaner, sample_raw_dataframe):
        """Test cleaning large dataset."""
        df = pd.concat([sample_raw_dataframe] * 100, ignore_index=True)
        result = cleaner.clean(df)
        assert len(result) > 0
