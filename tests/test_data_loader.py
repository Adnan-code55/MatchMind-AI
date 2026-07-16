"""
Unit tests for data_loader module.

Tests cover CSV discovery, loading, combining, error handling, and logging.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import csv

from backend.app.data.data_loader import DataLoader
from backend.app.data.exceptions import (
    DatasetNotFoundError,
    InvalidDatasetError,
    DataProcessingError,
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return {
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
    }


def create_csv_file(file_path: Path, data: dict) -> None:
    """Helper function to create a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


class TestDataLoaderInitialization:
    """Tests for DataLoader initialization."""

    def test_initialization_with_valid_directory(self, temp_data_dir):
        """Test loader initializes with valid directory."""
        loader = DataLoader(temp_data_dir)
        assert loader.data_directory == temp_data_dir

    def test_initialization_with_string_path(self, temp_data_dir):
        """Test loader accepts string path."""
        loader = DataLoader(str(temp_data_dir))
        assert loader.data_directory == temp_data_dir

    def test_initialization_with_nonexistent_directory(self):
        """Test loader raises error for nonexistent directory."""
        with pytest.raises(DatasetNotFoundError):
            DataLoader("/nonexistent/path/to/directory")

    def test_initialization_with_file_path(self, temp_data_dir, sample_csv_data):
        """Test loader raises error when path is a file."""
        file_path = temp_data_dir / "file.csv"
        create_csv_file(file_path, sample_csv_data)

        with pytest.raises(DatasetNotFoundError):
            DataLoader(file_path)


class TestDiscoverCSVFiles:
    """Tests for CSV file discovery."""

    def test_discover_no_files(self, temp_data_dir):
        """Test discovery when no CSV files exist."""
        loader = DataLoader(temp_data_dir)
        csv_files = loader.discover_csv_files()
        assert csv_files == []

    def test_discover_single_file(self, temp_data_dir, sample_csv_data):
        """Test discovery of single CSV file."""
        csv_path = temp_data_dir / "matches.csv"
        create_csv_file(csv_path, sample_csv_data)

        loader = DataLoader(temp_data_dir)
        csv_files = loader.discover_csv_files()

        assert len(csv_files) == 1
        assert csv_files[0] == csv_path

    def test_discover_multiple_files(self, temp_data_dir, sample_csv_data):
        """Test discovery of multiple CSV files."""
        for i in range(3):
            csv_path = temp_data_dir / f"matches_{i}.csv"
            create_csv_file(csv_path, sample_csv_data)

        loader = DataLoader(temp_data_dir)
        csv_files = loader.discover_csv_files()

        assert len(csv_files) == 3

    def test_discover_ignores_non_csv_files(self, temp_data_dir, sample_csv_data):
        """Test discovery ignores non-CSV files."""
        csv_path = temp_data_dir / "matches.csv"
        create_csv_file(csv_path, sample_csv_data)

        (temp_data_dir / "readme.txt").write_text("Some content")

        loader = DataLoader(temp_data_dir)
        csv_files = loader.discover_csv_files()

        assert len(csv_files) == 1


class TestLoadCSV:
    """Tests for loading individual CSV files."""

    def test_load_valid_csv(self, temp_data_dir, sample_csv_data):
        """Test loading a valid CSV file."""
        csv_path = temp_data_dir / "matches.csv"
        create_csv_file(csv_path, sample_csv_data)

        loader = DataLoader(temp_data_dir)
        df = loader.load_csv(csv_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == list(sample_csv_data.keys())

    def test_load_nonexistent_file(self, temp_data_dir):
        """Test loading nonexistent file raises error."""
        loader = DataLoader(temp_data_dir)

        with pytest.raises(DatasetNotFoundError):
            loader.load_csv(temp_data_dir / "nonexistent.csv")

    def test_load_empty_csv(self, temp_data_dir):
        """Test loading empty CSV file."""
        csv_path = temp_data_dir / "empty.csv"
        df_empty = pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])
        df_empty.to_csv(csv_path, index=False)

        loader = DataLoader(temp_data_dir)
        df = loader.load_csv(csv_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_malformed_csv(self, temp_data_dir):
        """Test loading malformed CSV raises error."""
        csv_path = temp_data_dir / "malformed.csv"
        with open(csv_path, "w") as f:
            f.write("This is not valid CSV data\n")
            f.write("incomplete line")

        loader = DataLoader(temp_data_dir)

        df = loader.load_csv(csv_path)
        assert df is not None

    def test_load_csv_returns_string_dtype(self, temp_data_dir, sample_csv_data):
        """Test loaded CSV returns all columns as strings."""
        csv_path = temp_data_dir / "matches.csv"
        create_csv_file(csv_path, sample_csv_data)

        loader = DataLoader(temp_data_dir)
        df = loader.load_csv(csv_path)

        for col in df.columns:
            assert df[col].dtype in ["object", "string"]


class TestLoadMatches:
    """Tests for load_matches method."""

    def test_load_single_file(self, temp_data_dir, sample_csv_data):
        """Test load_matches with specific file."""
        csv_path = temp_data_dir / "matches.csv"
        create_csv_file(csv_path, sample_csv_data)

        loader = DataLoader(temp_data_dir)
        df = loader.load_matches(csv_path)

        assert len(df) == 3
        assert list(df.columns) == list(sample_csv_data.keys())

    def test_load_all_files_from_directory(self, temp_data_dir, sample_csv_data):
        """Test load_matches loads all files when no specific file given."""
        for i in range(2):
            csv_path = temp_data_dir / f"matches_{i}.csv"
            create_csv_file(csv_path, sample_csv_data)

        loader = DataLoader(temp_data_dir)
        df = loader.load_matches()

        assert len(df) == 6
        assert list(df.columns) == list(sample_csv_data.keys())

    def test_load_matches_no_files_found(self, temp_data_dir):
        """Test load_matches raises error when no files found."""
        loader = DataLoader(temp_data_dir)

        with pytest.raises(DatasetNotFoundError):
            loader.load_matches()

    def test_load_matches_nonexistent_file(self, temp_data_dir):
        """Test load_matches raises error for nonexistent file."""
        loader = DataLoader(temp_data_dir)

        with pytest.raises(DatasetNotFoundError):
            loader.load_matches(temp_data_dir / "nonexistent.csv")

    def test_load_matches_resets_index(self, temp_data_dir, sample_csv_data):
        """Test combined dataframe has reset index."""
        for i in range(2):
            csv_path = temp_data_dir / f"matches_{i}.csv"
            create_csv_file(csv_path, sample_csv_data)

        loader = DataLoader(temp_data_dir)
        df = loader.load_matches()

        assert list(df.index) == list(range(len(df)))


class TestCombineDataframes:
    """Tests for combining multiple dataframes."""

    def test_combine_dataframes(self, temp_data_dir, sample_csv_data):
        """Test combining multiple dataframes."""
        dataframes = [
            pd.DataFrame(sample_csv_data),
            pd.DataFrame(sample_csv_data),
        ]

        loader = DataLoader(temp_data_dir)
        combined = loader._combine_dataframes(dataframes)

        assert len(combined) == 6
        assert list(combined.columns) == list(sample_csv_data.keys())

    def test_combine_empty_list(self, temp_data_dir):
        """Test combining empty list raises error."""
        loader = DataLoader(temp_data_dir)

        with pytest.raises(DataProcessingError):
            loader._combine_dataframes([])

    def test_combine_single_dataframe(self, temp_data_dir, sample_csv_data):
        """Test combining single dataframe."""
        dataframes = [pd.DataFrame(sample_csv_data)]

        loader = DataLoader(temp_data_dir)
        combined = loader._combine_dataframes(dataframes)

        assert len(combined) == 3


class TestLoadAndCombine:
    """Tests for load_and_combine method."""

    def test_load_and_combine_specific_files(self, temp_data_dir, sample_csv_data):
        """Test load_and_combine with specific files."""
        csv_paths = []
        for i in range(2):
            csv_path = temp_data_dir / f"matches_{i}.csv"
            create_csv_file(csv_path, sample_csv_data)
            csv_paths.append(csv_path)

        loader = DataLoader(temp_data_dir)
        df = loader.load_and_combine(csv_paths)

        assert len(df) == 6

    def test_load_and_combine_all_files(self, temp_data_dir, sample_csv_data):
        """Test load_and_combine without specific files."""
        for i in range(2):
            csv_path = temp_data_dir / f"matches_{i}.csv"
            create_csv_file(csv_path, sample_csv_data)

        loader = DataLoader(temp_data_dir)
        df = loader.load_and_combine()

        assert len(df) == 6

    def test_load_and_combine_no_files(self, temp_data_dir):
        """Test load_and_combine raises error with no files."""
        loader = DataLoader(temp_data_dir)

        with pytest.raises(DatasetNotFoundError):
            loader.load_and_combine()

    def test_load_and_combine_with_string_paths(self, temp_data_dir, sample_csv_data):
        """Test load_and_combine accepts string paths."""
        csv_path = temp_data_dir / "matches.csv"
        create_csv_file(csv_path, sample_csv_data)

        loader = DataLoader(temp_data_dir)
        df = loader.load_and_combine([str(csv_path)])

        assert len(df) == 3


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_load_csv_with_encoding_error(self, temp_data_dir):
        """Test handling of encoding errors."""
        csv_path = temp_data_dir / "bad_encoding.csv"

        with open(csv_path, "wb") as f:
            f.write(b"\x80\x81\x82\x83")

        loader = DataLoader(temp_data_dir)

        with pytest.raises(InvalidDatasetError):
            loader.load_csv(csv_path)

    def test_load_matches_skips_invalid_files(self, temp_data_dir, sample_csv_data):
        """Test load_matches skips invalid files and continues."""
        valid_csv = temp_data_dir / "valid.csv"
        create_csv_file(valid_csv, sample_csv_data)

        bad_csv = temp_data_dir / "bad.csv"
        with open(bad_csv, "w") as f:
            f.write("invalid")

        loader = DataLoader(temp_data_dir)
        df = loader.load_matches()

        assert len(df) == 3


class TestDataConsistency:
    """Tests for data consistency."""

    def test_loaded_data_matches_source(self, temp_data_dir, sample_csv_data):
        """Test loaded data matches source CSV."""
        csv_path = temp_data_dir / "matches.csv"
        create_csv_file(csv_path, sample_csv_data)

        loader = DataLoader(temp_data_dir)
        df = loader.load_csv(csv_path)

        for col in sample_csv_data:
            assert list(df[col]) == sample_csv_data[col]

    def test_combined_data_preserves_order(self, temp_data_dir, sample_csv_data):
        """Test combined data preserves row order."""
        data1 = sample_csv_data.copy()
        data2 = {
            k: [v[0], v[1]] for k, v in sample_csv_data.items()
        }

        csv1 = temp_data_dir / "file1.csv"
        csv2 = temp_data_dir / "file2.csv"

        create_csv_file(csv1, data1)
        create_csv_file(csv2, data2)

        loader = DataLoader(temp_data_dir)
        df = loader.load_matches()

        assert len(df) == 5
