"""
Data cleaner module for MatchMind AI pipeline.

Responsible for cleaning and preparing raw data. Removes duplicates,
standardizes team names, converts dates, fills missing values, converts
data types, and sorts data chronologically.
"""

from typing import Dict, Optional, Set
import pandas as pd
import numpy as np

from .logger import PipelineLogger
from .exceptions import DataProcessingError
from .schema import FootballMatchSchema


MODULE_NAME = "cleaner"


class DataCleaner:
    """
    Cleaner for football match datasets.

    Performs data cleaning operations including duplicate removal, standardization,
    type conversion, and chronological sorting. All operations are applied with
    proper error handling and logging.
    """

    def __init__(self) -> None:
        """Initialize the data cleaner."""
        self.schema = FootballMatchSchema
        PipelineLogger.log_info(MODULE_NAME, "DataCleaner initialized")
        self._team_name_mapping: Dict[str, str] = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform all cleaning operations on dataset.

        Args:
            df (pd.DataFrame): Raw dataset to clean.

        Returns:
            pd.DataFrame: Cleaned dataset.

        Raises:
            DataProcessingError: If cleaning operations fail.
        """
        try:
            original_rows = len(df)

            df = self.remove_duplicates(df)
            df = self.standardize_team_names(df)
            df = self.convert_dates(df)
            df = self.convert_data_types(df)
            df = self.fill_missing_values(df)
            df = self.sort_chronologically(df)

            cleaned_rows = len(df)
            removed_rows = original_rows - cleaned_rows

            PipelineLogger.log_info(
                MODULE_NAME,
                f"Cleaning complete. Rows: {original_rows} → {cleaned_rows} "
                f"({removed_rows} removed)",
            )

            return df

        except Exception as e:
            message = f"Unexpected error during cleaning: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from dataset.

        Args:
            df (pd.DataFrame): Dataset potentially containing duplicates.

        Returns:
            pd.DataFrame: Dataset with duplicates removed.
        """
        try:
            initial_count = len(df)
            df_cleaned = df.drop_duplicates(keep="first").reset_index(drop=True)
            removed_count = initial_count - len(df_cleaned)

            if removed_count > 0:
                PipelineLogger.log_info(
                    MODULE_NAME,
                    f"Removed {removed_count} duplicate row(s)",
                )

            return df_cleaned

        except Exception as e:
            message = f"Error removing duplicates: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def standardize_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize team names to consistent format.

        Removes leading/trailing whitespace, converts to title case for consistency.

        Args:
            df (pd.DataFrame): Dataset with team names to standardize.

        Returns:
            pd.DataFrame: Dataset with standardized team names.
        """
        try:
            df = df.copy()

            for col in ["HomeTeam", "AwayTeam"]:
                if col not in df.columns:
                    continue

                df[col] = df[col].astype(str).str.strip()

            PipelineLogger.log_debug(
                MODULE_NAME,
                "Team names standardized",
            )

            return df

        except Exception as e:
            message = f"Error standardizing team names: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date column to datetime format.

        Attempts to parse dates with flexible format detection. Converts Date
        column to datetime64[ns] type.

        Args:
            df (pd.DataFrame): Dataset with date column.

        Returns:
            pd.DataFrame: Dataset with converted dates.

        Raises:
            DataProcessingError: If date conversion fails.
        """
        try:
            df = df.copy()

            if "Date" not in df.columns:
                return df

            try:
                df["Date"] = pd.to_datetime(df["Date"], format="mixed", infer_datetime_format=True)
            except Exception:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            PipelineLogger.log_debug(
                MODULE_NAME,
                "Date column converted to datetime64[ns]",
            )

            return df

        except Exception as e:
            message = f"Error converting dates: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types to match schema.

        Converts numeric columns to int64, categorical columns to string,
        and ensures dates are datetime64[ns].

        Args:
            df (pd.DataFrame): Dataset with types to convert.

        Returns:
            pd.DataFrame: Dataset with proper data types.

        Raises:
            DataProcessingError: If type conversion fails.
        """
        try:
            df = df.copy()
            dtypes = self.schema.get_all_dtypes()

            for col, target_dtype in dtypes.items():
                if col not in df.columns:
                    continue

                if target_dtype == "datetime64[ns]":
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col], errors="coerce")

                elif target_dtype == "int64":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    df[col] = df[col].fillna(0).astype("int64")

                elif target_dtype == "float64":
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                elif target_dtype == "object":
                    df[col] = df[col].astype(str)

            PipelineLogger.log_debug(
                MODULE_NAME,
                f"Data types converted for {len(dtypes)} columns",
            )

            return df

        except Exception as e:
            message = f"Error converting data types: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill or remove rows with missing values.

        For numeric columns, fills with 0. For string columns, fills with "Unknown".
        Removes rows where critical columns (Date, team names) are still null.

        Args:
            df (pd.DataFrame): Dataset with potential missing values.

        Returns:
            pd.DataFrame: Dataset with missing values handled.
        """
        try:
            df = df.copy()
            initial_rows = len(df)

            numeric_cols = self.schema.get_numeric_columns()
            for col in numeric_cols:
                if col in df.columns and df[col].isnull().any():
                    df[col] = df[col].fillna(0)

            categorical_cols = self.schema.get_categorical_columns()
            for col in categorical_cols:
                if col in df.columns and df[col].isnull().any():
                    df[col] = df[col].fillna("Unknown")

            df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

            final_rows = len(df)
            removed_rows = initial_rows - final_rows

            if removed_rows > 0:
                PipelineLogger.log_info(
                    MODULE_NAME,
                    f"Removed {removed_rows} row(s) with critical missing values",
                )

            return df.reset_index(drop=True)

        except Exception as e:
            message = f"Error filling missing values: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort dataset chronologically by date.

        Sorts by Date column in ascending order, then resets index.

        Args:
            df (pd.DataFrame): Dataset to sort.

        Returns:
            pd.DataFrame: Dataset sorted by date.

        Raises:
            DataProcessingError: If sorting fails.
        """
        try:
            df = df.copy()

            if "Date" not in df.columns:
                return df

            df = df.sort_values("Date", ascending=True)
            df = df.reset_index(drop=True)

            PipelineLogger.log_debug(
                MODULE_NAME,
                "Dataset sorted chronologically by date",
            )

            return df

        except Exception as e:
            message = f"Error sorting chronologically: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def register_team_name_mapping(
        self, mapping: Dict[str, str]
    ) -> None:
        """
        Register custom team name mappings.

        Allows mapping of team names to standard names for consolidation.

        Args:
            mapping (Dict[str, str]): Dictionary mapping original names to standard names.
        """
        self._team_name_mapping.update(mapping)
        PipelineLogger.log_debug(
            MODULE_NAME,
            f"Registered {len(mapping)} team name mapping(s)",
        )

    def apply_team_name_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply custom team name mappings to dataset.

        Args:
            df (pd.DataFrame): Dataset with team names to map.

        Returns:
            pd.DataFrame: Dataset with mapped team names.
        """
        if not self._team_name_mapping:
            return df

        try:
            df = df.copy()

            for col in ["HomeTeam", "AwayTeam"]:
                if col not in df.columns:
                    continue

                df[col] = df[col].map(
                    lambda x: self._team_name_mapping.get(x, x)
                )

            PipelineLogger.log_debug(
                MODULE_NAME,
                "Team name mappings applied",
            )

            return df

        except Exception as e:
            message = f"Error applying team name mapping: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e
