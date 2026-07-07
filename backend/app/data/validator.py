"""
Data validator module for MatchMind AI pipeline.

Responsible for validating datasets against schema requirements. Validates
required columns, detects duplicates, identifies null values, and validates
data types, dates, scores, and team names.
"""

from typing import Dict, List, Set, Tuple, Any
import pandas as pd
import re

from .logger import PipelineLogger
from .exceptions import (
    DataValidationError,
    MissingColumnError,
    DuplicateRowError,
    NullValueError,
    InvalidDateError,
    InvalidScoreError,
    InvalidTeamNameError,
    SchemaMismatchError,
)
from .schema import FootballMatchSchema


MODULE_NAME = "validator"


class ValidationReport:
    """
    Container for dataset validation results.

    Stores all validation findings including errors, warnings, and summary
    statistics for comprehensive validation reporting.
    """

    def __init__(self) -> None:
        """Initialize an empty validation report."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.is_valid: bool = True
        self.checks_performed: Dict[str, bool] = {}

    def add_error(self, error_message: str) -> None:
        """
        Add error to report.

        Args:
            error_message (str): Error message to add.
        """
        self.errors.append(error_message)
        self.is_valid = False

    def add_warning(self, warning_message: str) -> None:
        """
        Add warning to report.

        Args:
            warning_message (str): Warning message to add.
        """
        self.warnings.append(warning_message)

    def add_check(self, check_name: str, passed: bool) -> None:
        """
        Record result of a validation check.

        Args:
            check_name (str): Name of the check performed.
            passed (bool): Whether the check passed.
        """
        self.checks_performed[check_name] = passed
        if not passed:
            self.is_valid = False

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation results.

        Returns:
            Dict[str, Any]: Summary dictionary with validation results.
        """
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "checks_performed": self.checks_performed,
        }


class DataValidator:
    """
    Validator for football match datasets.

    Validates datasets against the defined schema and performs comprehensive
    quality checks on data values, formats, and integrity.
    """

    def __init__(self) -> None:
        """Initialize the data validator."""
        self.schema = FootballMatchSchema
        PipelineLogger.log_info(MODULE_NAME, "DataValidator initialized")

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Perform comprehensive validation on dataset.

        Args:
            df (pd.DataFrame): Dataset to validate.

        Returns:
            ValidationReport: Detailed validation report.

        Raises:
            DataValidationError: If validation fails critically.
        """
        report = ValidationReport()

        try:
            self._validate_schema(df, report)
            self._validate_duplicates(df, report)
            self._validate_null_values(df, report)
            self._validate_dates(df, report)
            self._validate_scores(df, report)
            self._validate_team_names(df, report)

            if not report.is_valid:
                summary = report.get_summary()
                PipelineLogger.log_error(
                    MODULE_NAME,
                    f"Validation failed with {summary['error_count']} error(s)",
                )
                raise DataValidationError(
                    f"Validation failed: {len(report.errors)} error(s)"
                )

            PipelineLogger.log_info(
                MODULE_NAME,
                f"Validation successful. {len(df)} rows validated.",
            )
            return report

        except DataValidationError:
            raise
        except Exception as e:
            message = f"Unexpected error during validation: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            report.add_error(message)
            raise DataValidationError(message) from e

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate dataset schema against defined schema.

        Args:
            df (pd.DataFrame): Dataset to validate.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)

        Raises:
            SchemaMismatchError: If schema validation fails.
        """
        report = ValidationReport()
        self._validate_schema(df, report)

        if report.errors:
            raise SchemaMismatchError(
                f"Schema validation failed: {'; '.join(report.errors)}"
            )

        return True, []

    def _validate_schema(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """
        Check that all required columns exist.

        Args:
            df (pd.DataFrame): Dataset to validate.
            report (ValidationReport): Report to add findings to.

        Raises:
            MissingColumnError: If required columns are missing.
        """
        required_columns = set(self.schema.get_required_columns())
        existing_columns = set(df.columns)
        missing_columns = required_columns - existing_columns

        if missing_columns:
            message = f"Missing required columns: {', '.join(sorted(missing_columns))}"
            report.add_error(message)
            PipelineLogger.log_error(MODULE_NAME, message)
            raise MissingColumnError(message)

        report.add_check("schema_validation", True)

    def _validate_duplicates(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """
        Check for duplicate rows in dataset.

        Args:
            df (pd.DataFrame): Dataset to validate.
            report (ValidationReport): Report to add findings to.

        Raises:
            DuplicateRowError: If duplicates are found.
        """
        duplicates = df.duplicated().sum()

        if duplicates > 0:
            message = f"Found {duplicates} duplicate row(s)"
            report.add_error(message)
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DuplicateRowError(message)

        report.add_check("duplicate_check", True)

    def _validate_null_values(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """
        Check for null/missing values in dataset.

        Args:
            df (pd.DataFrame): Dataset to validate.
            report (ValidationReport): Report to add findings to.

        Raises:
            NullValueError: If null values are found.
        """
        null_counts = df.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]

        if not columns_with_nulls.empty:
            null_summary = {col: int(count) for col, count in columns_with_nulls.items()}
            message = f"Found null values in columns: {null_summary}"
            report.add_error(message)
            PipelineLogger.log_error(MODULE_NAME, message)
            raise NullValueError(message)

        report.add_check("null_value_check", True)

    def _validate_dates(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """
        Validate date columns contain valid date values.

        Args:
            df (pd.DataFrame): Dataset to validate.
            report (ValidationReport): Report to add findings to.

        Raises:
            InvalidDateError: If invalid dates are found.
        """
        if "Date" not in df.columns:
            report.add_check("date_validation", True)
            return

        invalid_dates: List[str] = []

        for idx, date_str in enumerate(df["Date"]):
            if pd.isna(date_str):
                continue

            try:
                pd.to_datetime(date_str)
            except (ValueError, TypeError):
                invalid_dates.append(f"Row {idx}: '{date_str}'")
                if len(invalid_dates) >= 5:
                    break

        if invalid_dates:
            message = f"Found {len(invalid_dates)} invalid date(s): {'; '.join(invalid_dates[:3])}"
            report.add_error(message)
            PipelineLogger.log_error(MODULE_NAME, message)
            raise InvalidDateError(message)

        report.add_check("date_validation", True)

    def _validate_scores(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """
        Validate score columns contain valid non-negative numeric values.

        Args:
            df (pd.DataFrame): Dataset to validate.
            report (ValidationReport): Report to add findings to.

        Raises:
            InvalidScoreError: If invalid scores are found.
        """
        score_columns = {"FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"}
        score_columns = score_columns & set(df.columns)

        invalid_scores: List[str] = []

        for col in score_columns:
            for idx, val in enumerate(df[col]):
                if pd.isna(val):
                    continue

                try:
                    score = int(val)
                    if score < 0:
                        invalid_scores.append(f"{col}[{idx}]: {val} (negative)")
                        if len(invalid_scores) >= 5:
                            break
                except (ValueError, TypeError):
                    invalid_scores.append(f"{col}[{idx}]: '{val}' (non-numeric)")
                    if len(invalid_scores) >= 5:
                        break

            if len(invalid_scores) >= 5:
                break

        if invalid_scores:
            message = f"Found {len(invalid_scores)} invalid score(s): {'; '.join(invalid_scores[:3])}"
            report.add_error(message)
            PipelineLogger.log_error(MODULE_NAME, message)
            raise InvalidScoreError(message)

        report.add_check("score_validation", True)

    def _validate_team_names(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """
        Validate team name columns contain valid non-empty strings.

        Args:
            df (pd.DataFrame): Dataset to validate.
            report (ValidationReport): Report to add findings to.

        Raises:
            InvalidTeamNameError: If invalid team names are found.
        """
        team_columns = {"HomeTeam", "AwayTeam"}
        team_columns = team_columns & set(df.columns)

        invalid_teams: List[str] = []

        for col in team_columns:
            for idx, team in enumerate(df[col]):
                if pd.isna(team):
                    continue

                team_str = str(team).strip()

                if not team_str:
                    invalid_teams.append(f"{col}[{idx}]: empty string")
                elif len(team_str) > 50:
                    invalid_teams.append(f"{col}[{idx}]: '{team_str[:20]}...' (too long)")
                elif not re.match(r"^[a-zA-Z0-9\s\-'&.]+$", team_str):
                    invalid_teams.append(f"{col}[{idx}]: '{team_str}' (invalid characters)")

                if len(invalid_teams) >= 5:
                    break

            if len(invalid_teams) >= 5:
                break

        if invalid_teams:
            message = f"Found {len(invalid_teams)} invalid team name(s): {'; '.join(invalid_teams[:3])}"
            report.add_error(message)
            PipelineLogger.log_error(MODULE_NAME, message)
            raise InvalidTeamNameError(message)

        report.add_check("team_name_validation", True)
