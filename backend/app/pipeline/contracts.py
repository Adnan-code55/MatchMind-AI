"""
Data contracts for MatchMind AI.

This module defines business rule contracts that validate football matches for
production readiness before the pipeline proceeds.
"""

from datetime import datetime
from typing import List

import pandas as pd

from ..data.exceptions import (
    DuplicateRowError,
    InvalidDateError,
    InvalidScoreError,
    MissingColumnError,
)
from ..data.schema import FootballMatchSchema


class ContractValidator:
    """Validates a dataset against enterprise data contracts."""

    def __init__(self) -> None:
        """Initialize the contract validator."""
        self.schema = FootballMatchSchema

    def validate(self, df: pd.DataFrame) -> None:
        """Run all configured data contract checks."""
        self._validate_required_columns(df)
        self._validate_no_future_dates(df)
        self._validate_non_negative_scores(df)
        self._validate_correct_result_labels(df)
        self._validate_no_duplicate_matches(df)
        self._validate_data_types(df)

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Validate that required schema columns are present."""
        required_columns = set(self.schema.get_required_columns())
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise MissingColumnError(
                f"Required columns missing: {', '.join(sorted(missing_columns))}"
            )

    def _validate_no_future_dates(self, df: pd.DataFrame) -> None:
        """Validate that no match dates occur in the future."""
        if "Date" not in df.columns:
            return

        parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
        invalid_dates = parsed_dates[parsed_dates.isna()]
        if not invalid_dates.empty:
            row_indices = invalid_dates.index.tolist()[:5]
            raise InvalidDateError(
                f"Invalid dates found at rows: {row_indices}"
            )

        future_dates = parsed_dates[parsed_dates > datetime.now()]
        if not future_dates.empty:
            raise InvalidDateError(
                f"Future match dates detected in {len(future_dates)} row(s)"
            )

    def _validate_non_negative_scores(self, df: pd.DataFrame) -> None:
        """Validate that all goal counts are non-negative."""
        for column in ["FTHG", "FTAG"]:
            if column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            negative_values = values[values < 0]
            if not negative_values.empty:
                raise InvalidScoreError(
                    f"Negative values found in {column}: {negative_values.tolist()[:5]}"
                )

    def _validate_correct_result_labels(self, df: pd.DataFrame) -> None:
        """Validate that match result labels are consistent with scores."""
        if "FTR" not in df.columns or "FTHG" not in df.columns or "FTAG" not in df.columns:
            return

        valid_labels = {"H", "D", "A"}
        invalid_labels = df[~df["FTR"].isin(valid_labels)]
        if not invalid_labels.empty:
            raise InvalidScoreError(
                f"Invalid result labels found: {sorted(set(invalid_labels['FTR'].dropna().tolist()))}"
            )

        mismatches: List[str] = []
        for idx, row in df.iterrows():
            try:
                home_goals = int(row["FTHG"])
                away_goals = int(row["FTAG"])
            except (TypeError, ValueError):
                continue

            expected_label = "D" if home_goals == away_goals else ("H" if home_goals > away_goals else "A")
            if str(row["FTR"]).strip() != expected_label:
                mismatches.append(
                    f"row {idx}: expected {expected_label} but got {row['FTR']}"
                )

        if mismatches:
            raise InvalidScoreError(
                f"Result label mismatches: {mismatches[:5]}"
            )

    def _validate_no_duplicate_matches(self, df: pd.DataFrame) -> None:
        """Validate that the dataset contains no duplicate matches."""
        if not {"Date", "HomeTeam", "AwayTeam"}.issubset(df.columns):
            return

        duplicate_matches = df.duplicated(subset=["Date", "HomeTeam", "AwayTeam"], keep=False)
        if duplicate_matches.any():
            count = int(duplicate_matches.sum())
            raise DuplicateRowError(
                f"Duplicate match entries found: {count} row(s)"
            )

    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate that column values conform to expected schema types."""
        for column, dtype in self.schema.get_all_dtypes().items():
            if column not in df.columns:
                continue
            if dtype == "datetime64[ns]":
                parsed = pd.to_datetime(df[column], errors="coerce")
                if parsed.isna().any():
                    raise InvalidDateError(
                        f"Column {column} contains invalid datetime values"
                    )
            elif dtype == "int64":
                parsed = pd.to_numeric(df[column], errors="coerce")
                if parsed.isna().any():
                    raise InvalidScoreError(
                        f"Column {column} contains invalid integer values"
                    )
