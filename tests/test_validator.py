"""
Unit tests for validator module.

Tests cover schema validation, duplicate detection, null value checking,
date validation, score validation, and team name validation.
"""

import pytest
import pandas as pd
from datetime import datetime

from backend.app.data.validator import DataValidator, ValidationReport
from backend.app.data.exceptions import (
    DataValidationError,
    MissingColumnError,
    DuplicateRowError,
    NullValueError,
    InvalidDateError,
    InvalidScoreError,
    InvalidTeamNameError,
    SchemaMismatchError,
)


@pytest.fixture
def sample_valid_dataframe():
    """Create a sample valid dataframe for testing."""
    return pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "HomeTeam": ["Arsenal", "Manchester United", "Liverpool"],
        "AwayTeam": ["Chelsea", "Manchester City", "Tottenham"],
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
def validator():
    """Create a DataValidator instance."""
    return DataValidator()


class TestValidationReport:
    """Tests for ValidationReport class."""

    def test_report_initialization(self):
        """Test ValidationReport initializes with correct state."""
        report = ValidationReport()
        assert report.is_valid is True
        assert report.errors == []
        assert report.warnings == []
        assert report.checks_performed == {}

    def test_add_error_marks_invalid(self):
        """Test adding error marks report as invalid."""
        report = ValidationReport()
        report.add_error("Test error")
        assert report.is_valid is False
        assert len(report.errors) == 1

    def test_add_warning_keeps_valid(self):
        """Test adding warning doesn't affect validity."""
        report = ValidationReport()
        report.add_warning("Test warning")
        assert report.is_valid is True
        assert len(report.warnings) == 1

    def test_add_check(self):
        """Test adding check to report."""
        report = ValidationReport()
        report.add_check("test_check", True)
        assert report.checks_performed["test_check"] is True

    def test_add_check_failure_marks_invalid(self):
        """Test failed check marks report as invalid."""
        report = ValidationReport()
        report.add_check("test_check", False)
        assert report.is_valid is False

    def test_get_summary(self):
        """Test getting summary from report."""
        report = ValidationReport()
        report.add_error("Error 1")
        report.add_warning("Warning 1")
        report.add_check("check1", True)

        summary = report.get_summary()

        assert summary["is_valid"] is False
        assert summary["error_count"] == 1
        assert summary["warning_count"] == 1
        assert "check1" in summary["checks_performed"]


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_validate_valid_schema(self, validator, sample_valid_dataframe):
        """Test validation passes with correct schema."""
        report = ValidationReport()
        validator._validate_schema(sample_valid_dataframe, report)
        assert report.errors == []

    def test_validate_missing_single_column(self, validator, sample_valid_dataframe):
        """Test validation fails with missing column."""
        df = sample_valid_dataframe.drop("FTHG", axis=1)
        report = ValidationReport()

        with pytest.raises(MissingColumnError):
            validator._validate_schema(df, report)

    def test_validate_missing_multiple_columns(self, validator, sample_valid_dataframe):
        """Test validation fails with multiple missing columns."""
        df = sample_valid_dataframe.drop(["FTHG", "FTAG", "FTR"], axis=1)
        report = ValidationReport()

        with pytest.raises(MissingColumnError):
            validator._validate_schema(df, report)

    def test_validate_schema_raises_error(self, validator, sample_valid_dataframe):
        """Test validate_schema method raises SchemaMismatchError."""
        df = sample_valid_dataframe.drop("FTHG", axis=1)

        with pytest.raises(SchemaMismatchError):
            validator.validate_schema(df)


class TestDuplicateValidation:
    """Tests for duplicate row validation."""

    def test_validate_no_duplicates(self, validator, sample_valid_dataframe):
        """Test validation passes with no duplicates."""
        report = ValidationReport()
        validator._validate_duplicates(sample_valid_dataframe, report)
        assert report.errors == []

    def test_validate_with_duplicates(self, validator, sample_valid_dataframe):
        """Test validation fails with duplicate rows."""
        df = pd.concat([sample_valid_dataframe, sample_valid_dataframe.iloc[[0]]])
        report = ValidationReport()

        with pytest.raises(DuplicateRowError):
            validator._validate_duplicates(df, report)

    def test_duplicate_error_message(self, validator, sample_valid_dataframe):
        """Test duplicate error message includes count."""
        df = pd.concat([sample_valid_dataframe, sample_valid_dataframe.iloc[[0, 1]]])
        report = ValidationReport()

        with pytest.raises(DuplicateRowError) as exc_info:
            validator._validate_duplicates(df, report)

        assert "duplicate" in str(exc_info.value).lower()


class TestNullValueValidation:
    """Tests for null value validation."""

    def test_validate_no_nulls(self, validator, sample_valid_dataframe):
        """Test validation passes with no null values."""
        report = ValidationReport()
        validator._validate_null_values(sample_valid_dataframe, report)
        assert report.errors == []

    def test_validate_with_nulls_in_column(self, validator, sample_valid_dataframe):
        """Test validation fails with null values."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "FTHG"] = None
        report = ValidationReport()

        with pytest.raises(NullValueError):
            validator._validate_null_values(df, report)

    def test_validate_with_multiple_nulls(self, validator, sample_valid_dataframe):
        """Test validation detects multiple null values."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "FTHG"] = None
        df.loc[1, "FTAG"] = None
        report = ValidationReport()

        with pytest.raises(NullValueError):
            validator._validate_null_values(df, report)

    def test_null_error_message_includes_columns(self, validator, sample_valid_dataframe):
        """Test null error message includes affected columns."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "FTHG"] = None
        report = ValidationReport()

        with pytest.raises(NullValueError) as exc_info:
            validator._validate_null_values(df, report)

        assert "FTHG" in str(exc_info.value)


class TestDateValidation:
    """Tests for date validation."""

    def test_validate_valid_dates(self, validator, sample_valid_dataframe):
        """Test validation passes with valid dates."""
        report = ValidationReport()
        validator._validate_dates(sample_valid_dataframe, report)
        assert report.errors == []

    def test_validate_invalid_date_format(self, validator, sample_valid_dataframe):
        """Test validation fails with invalid date format."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "Date"] = "invalid-date"
        report = ValidationReport()

        with pytest.raises(InvalidDateError):
            validator._validate_dates(df, report)

    def test_validate_multiple_invalid_dates(self, validator, sample_valid_dataframe):
        """Test validation detects multiple invalid dates."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "Date"] = "invalid1"
        df.loc[1, "Date"] = "invalid2"
        report = ValidationReport()

        with pytest.raises(InvalidDateError):
            validator._validate_dates(df, report)

    def test_validate_dates_without_date_column(self, validator, sample_valid_dataframe):
        """Test validation passes when Date column missing."""
        df = sample_valid_dataframe.drop("Date", axis=1)
        report = ValidationReport()
        validator._validate_dates(df, report)
        assert report.errors == []

    def test_date_error_message_includes_example(self, validator, sample_valid_dataframe):
        """Test date error message includes invalid date example."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "Date"] = "not-a-date"
        report = ValidationReport()

        with pytest.raises(InvalidDateError) as exc_info:
            validator._validate_dates(df, report)

        assert "invalid" in str(exc_info.value).lower()


class TestScoreValidation:
    """Tests for score validation."""

    def test_validate_valid_scores(self, validator, sample_valid_dataframe):
        """Test validation passes with valid scores."""
        report = ValidationReport()
        validator._validate_scores(sample_valid_dataframe, report)
        assert report.errors == []

    def test_validate_negative_score(self, validator, sample_valid_dataframe):
        """Test validation fails with negative score."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "FTHG"] = "-1"
        report = ValidationReport()

        with pytest.raises(InvalidScoreError):
            validator._validate_scores(df, report)

    def test_validate_non_numeric_score(self, validator, sample_valid_dataframe):
        """Test validation fails with non-numeric score."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "FTHG"] = "abc"
        report = ValidationReport()

        with pytest.raises(InvalidScoreError):
            validator._validate_scores(df, report)

    def test_validate_multiple_invalid_scores(self, validator, sample_valid_dataframe):
        """Test validation detects multiple invalid scores."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "FTHG"] = "invalid"
        df.loc[1, "FTAG"] = "-5"
        report = ValidationReport()

        with pytest.raises(InvalidScoreError):
            validator._validate_scores(df, report)

    def test_validate_zero_scores(self, validator, sample_valid_dataframe):
        """Test validation passes with zero scores."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "FTHG"] = "0"
        report = ValidationReport()
        validator._validate_scores(df, report)
        assert report.errors == []


class TestTeamNameValidation:
    """Tests for team name validation."""

    def test_validate_valid_team_names(self, validator, sample_valid_dataframe):
        """Test validation passes with valid team names."""
        report = ValidationReport()
        validator._validate_team_names(sample_valid_dataframe, report)
        assert report.errors == []

    def test_validate_empty_team_name(self, validator, sample_valid_dataframe):
        """Test validation fails with empty team name."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "HomeTeam"] = ""
        report = ValidationReport()

        with pytest.raises(InvalidTeamNameError):
            validator._validate_team_names(df, report)

    def test_validate_whitespace_only_team_name(self, validator, sample_valid_dataframe):
        """Test validation fails with whitespace-only team name."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "HomeTeam"] = "   "
        report = ValidationReport()

        with pytest.raises(InvalidTeamNameError):
            validator._validate_team_names(df, report)

    def test_validate_too_long_team_name(self, validator, sample_valid_dataframe):
        """Test validation fails with excessively long team name."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "HomeTeam"] = "A" * 100
        report = ValidationReport()

        with pytest.raises(InvalidTeamNameError):
            validator._validate_team_names(df, report)

    def test_validate_team_name_with_invalid_characters(self, validator, sample_valid_dataframe):
        """Test validation fails with invalid special characters."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "HomeTeam"] = "Arsenal@#$%"
        report = ValidationReport()

        with pytest.raises(InvalidTeamNameError):
            validator._validate_team_names(df, report)

    def test_validate_team_names_with_allowed_characters(self, validator, sample_valid_dataframe):
        """Test validation passes with allowed special characters."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "HomeTeam"] = "Saint-Etienne"
        df.loc[1, "AwayTeam"] = "Manchester United"
        df.loc[2, "HomeTeam"] = "Brighton & Hove"
        report = ValidationReport()
        validator._validate_team_names(df, report)
        assert report.errors == []


class TestComprehensiveValidation:
    """Tests for comprehensive validation."""

    def test_validate_complete_valid_dataset(self, validator, sample_valid_dataframe):
        """Test comprehensive validation passes on valid dataset."""
        report = validator.validate(sample_valid_dataframe)
        assert report.is_valid is True
        assert len(report.errors) == 0

    def test_validate_fails_on_missing_columns(self, validator, sample_valid_dataframe):
        """Test comprehensive validation fails on missing columns."""
        df = sample_valid_dataframe.drop("FTHG", axis=1)

        with pytest.raises(DataValidationError):
            validator.validate(df)

    def test_validate_fails_on_duplicates(self, validator, sample_valid_dataframe):
        """Test comprehensive validation fails on duplicates."""
        df = pd.concat([sample_valid_dataframe, sample_valid_dataframe.iloc[[0]]])

        with pytest.raises(DataValidationError):
            validator.validate(df)

    def test_validate_fails_on_nulls(self, validator, sample_valid_dataframe):
        """Test comprehensive validation fails on null values."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "FTHG"] = None

        with pytest.raises(DataValidationError):
            validator.validate(df)

    def test_validate_fails_on_invalid_dates(self, validator, sample_valid_dataframe):
        """Test comprehensive validation fails on invalid dates."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "Date"] = "not-a-date"

        with pytest.raises(DataValidationError):
            validator.validate(df)

    def test_validate_fails_on_invalid_scores(self, validator, sample_valid_dataframe):
        """Test comprehensive validation fails on invalid scores."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "FTHG"] = "-1"

        with pytest.raises(DataValidationError):
            validator.validate(df)

    def test_validate_fails_on_invalid_team_names(self, validator, sample_valid_dataframe):
        """Test comprehensive validation fails on invalid team names."""
        df = sample_valid_dataframe.copy()
        df.loc[0, "HomeTeam"] = ""

        with pytest.raises(DataValidationError):
            validator.validate(df)

    def test_validation_stops_at_first_error_group(self, validator, sample_valid_dataframe):
        """Test validation stops at first error group."""
        df = sample_valid_dataframe.drop("FTHG", axis=1)

        with pytest.raises(DataValidationError):
            validator.validate(df)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_validate_empty_dataframe(self, validator):
        """Test validation on empty dataframe."""
        df = pd.DataFrame(columns=[
            "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"
        ])

        report = validator.validate(df)
        assert report.is_valid is True

    def test_validate_single_row(self, validator, sample_valid_dataframe):
        """Test validation on single row."""
        df = sample_valid_dataframe.iloc[[0]]
        report = validator.validate(df)
        assert report.is_valid is True

    def test_validate_with_extra_columns(self, validator, sample_valid_dataframe):
        """Test validation passes with extra columns."""
        df = sample_valid_dataframe.copy()
        df["ExtraColumn"] = "extra_value"
        report = validator.validate(df)
        assert report.is_valid is True

    def test_validate_large_dataset(self, validator, sample_valid_dataframe):
        """Test validation on large dataset."""
        # Create a large dataset with unique rows by modifying Date or HomeTeam
        df = pd.concat([sample_valid_dataframe] * 100, ignore_index=True)
        df["Date"] = pd.date_range("2023-01-01", periods=len(df))
        report = validator.validate(df)
        assert report.is_valid is True
