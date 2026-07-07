"""
Data loader module for MatchMind AI pipeline.

Responsible for discovering, loading, and combining CSV files. Handles
missing files, invalid CSVs, and provides detailed logging of all operations.
"""

from pathlib import Path
from typing import Optional, Union, List
import pandas as pd

from .logger import PipelineLogger
from .exceptions import (
    DatasetNotFoundError,
    InvalidDatasetError,
    DataProcessingError,
)
from .schema import FootballMatchSchema


MODULE_NAME = "data_loader"


class DataLoader:
    """
    Loader for football match CSV data.

    Discovers CSV files in a directory, loads them individually or in batch,
    combines multiple datasets, and handles various error scenarios gracefully.
    """

    def __init__(self, data_directory: Union[str, Path]) -> None:
        """
        Initialize the data loader with a data directory.

        Args:
            data_directory (Union[str, Path]): Path to directory containing CSV files.

        Raises:
            DatasetNotFoundError: If directory does not exist.
        """
        self.data_directory = Path(data_directory)

        if not self.data_directory.exists():
            message = f"Data directory does not exist: {self.data_directory}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DatasetNotFoundError(message)

        if not self.data_directory.is_dir():
            message = f"Path is not a directory: {self.data_directory}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DatasetNotFoundError(message)

        PipelineLogger.log_info(
            MODULE_NAME,
            f"DataLoader initialized with directory: {self.data_directory}",
        )

    def discover_csv_files(self) -> List[Path]:
        """
        Discover all CSV files in the data directory.

        Returns:
            List[Path]: List of paths to CSV files found.
        """
        csv_files = list(self.data_directory.glob("*.csv"))
        PipelineLogger.log_info(
            MODULE_NAME,
            f"Discovered {len(csv_files)} CSV file(s) in {self.data_directory}",
        )
        return csv_files

    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load a single CSV file.

        Args:
            file_path (Union[str, Path]): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataset as DataFrame.

        Raises:
            DatasetNotFoundError: If file does not exist.
            InvalidDatasetError: If file cannot be read or is invalid.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            message = f"CSV file not found: {file_path}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DatasetNotFoundError(message)

        try:
            df = pd.read_csv(file_path, dtype=str)

            if df.empty:
                message = f"CSV file is empty: {file_path}"
                PipelineLogger.log_warning(MODULE_NAME, message)

            PipelineLogger.log_info(
                MODULE_NAME,
                f"Successfully loaded CSV: {file_path.name} ({len(df)} rows)",
            )
            return df

        except pd.errors.ParserError as e:
            message = f"Failed to parse CSV file {file_path}: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise InvalidDatasetError(message) from e

        except (UnicodeDecodeError, OSError) as e:
            message = f"Failed to read CSV file {file_path}: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise InvalidDatasetError(message) from e

        except Exception as e:
            message = f"Unexpected error loading CSV {file_path}: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise InvalidDatasetError(message) from e

    def load_matches(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Load match data from CSV file(s).

        If file_path is provided, loads that specific file. Otherwise, discovers
        and loads all CSV files from the data directory, combining them into
        a single DataFrame.

        Args:
            file_path (Optional[Union[str, Path]]): Specific CSV file to load.
                If None, loads all CSV files from directory. Defaults to None.

        Returns:
            pd.DataFrame: Combined dataset from all loaded files.

        Raises:
            DatasetNotFoundError: If no files found or specified file missing.
            InvalidDatasetError: If CSV files cannot be read.
            DataProcessingError: If combination of datasets fails.
        """
        try:
            if file_path:
                df = self.load_csv(file_path)
                return df

            csv_files = self.discover_csv_files()

            if not csv_files:
                message = f"No CSV files found in {self.data_directory}"
                PipelineLogger.log_error(MODULE_NAME, message)
                raise DatasetNotFoundError(message)

            dataframes: List[pd.DataFrame] = []

            for csv_file in csv_files:
                try:
                    df = self.load_csv(csv_file)
                    dataframes.append(df)
                except InvalidDatasetError:
                    message = f"Skipping invalid CSV: {csv_file.name}"
                    PipelineLogger.log_warning(MODULE_NAME, message)
                    continue

            if not dataframes:
                message = "No valid CSV files could be loaded"
                PipelineLogger.log_error(MODULE_NAME, message)
                raise DataProcessingError(message)

            combined_df = self._combine_dataframes(dataframes)
            return combined_df

        except (DatasetNotFoundError, InvalidDatasetError):
            raise
        except Exception as e:
            message = f"Unexpected error in load_matches: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def _combine_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple DataFrames into one.

        Args:
            dataframes (List[pd.DataFrame]): List of DataFrames to combine.

        Returns:
            pd.DataFrame: Combined DataFrame.

        Raises:
            DataProcessingError: If combination fails.
        """
        try:
            combined = pd.concat(dataframes, ignore_index=True)
            PipelineLogger.log_info(
                MODULE_NAME,
                f"Combined {len(dataframes)} dataset(s) into single DataFrame "
                f"with {len(combined)} rows",
            )
            return combined

        except Exception as e:
            message = f"Failed to combine dataframes: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e

    def load_and_combine(
        self, file_paths: Optional[List[Union[str, Path]]] = None
    ) -> pd.DataFrame:
        """
        Load and combine specific CSV files or all files in directory.

        Args:
            file_paths (Optional[List[Union[str, Path]]]): List of specific
                CSV files to load. If None, loads all files from directory.
                Defaults to None.

        Returns:
            pd.DataFrame: Combined dataset.

        Raises:
            DatasetNotFoundError: If no files found.
            InvalidDatasetError: If CSV files cannot be read.
            DataProcessingError: If combination fails.
        """
        try:
            if file_paths is None:
                csv_files = self.discover_csv_files()
                if not csv_files:
                    message = f"No CSV files found in {self.data_directory}"
                    PipelineLogger.log_error(MODULE_NAME, message)
                    raise DatasetNotFoundError(message)
                file_paths = csv_files
            else:
                file_paths = [Path(fp) for fp in file_paths]

            dataframes: List[pd.DataFrame] = []

            for file_path in file_paths:
                try:
                    df = self.load_csv(file_path)
                    dataframes.append(df)
                except InvalidDatasetError:
                    message = f"Skipping invalid CSV: {file_path.name}"
                    PipelineLogger.log_warning(MODULE_NAME, message)
                    continue

            if not dataframes:
                message = "No valid CSV files could be loaded"
                PipelineLogger.log_error(MODULE_NAME, message)
                raise DataProcessingError(message)

            combined_df = self._combine_dataframes(dataframes)
            return combined_df

        except (DatasetNotFoundError, InvalidDatasetError, DataProcessingError):
            raise
        except Exception as e:
            message = f"Unexpected error in load_and_combine: {str(e)}"
            PipelineLogger.log_error(MODULE_NAME, message)
            raise DataProcessingError(message) from e
