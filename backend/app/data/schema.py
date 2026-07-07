"""
Dataset schema definition for football match data.

This module defines the required columns, data types, and validation rules
for the MatchMind AI dataset. It provides a centralized source of truth for
the data structure across all modules.
"""

from typing import Dict, List, Set
from enum import Enum


class DataType(Enum):
    """Enumeration of supported data types in the schema."""

    DATE = "datetime64[ns]"
    STRING = "object"
    FLOAT = "float64"
    INTEGER = "int64"


class FootballMatchSchema:
    """
    Schema definition for football match data.

    This class defines all required columns, their data types, and validation
    rules. It provides methods to validate dataframes against the schema and
    retrieve schema information for data processing.
    """

    _COLUMNS: Dict[str, DataType] = {
        "Date": DataType.DATE,
        "HomeTeam": DataType.STRING,
        "AwayTeam": DataType.STRING,
        "FTHG": DataType.INTEGER,
        "FTAG": DataType.INTEGER,
        "FTR": DataType.STRING,
        "HS": DataType.INTEGER,
        "AS": DataType.INTEGER,
        "HST": DataType.INTEGER,
        "AST": DataType.INTEGER,
        "HC": DataType.INTEGER,
        "AC": DataType.INTEGER,
        "HY": DataType.INTEGER,
        "AY": DataType.INTEGER,
        "HR": DataType.INTEGER,
        "AR": DataType.INTEGER,
    }

    _OPTIONAL_COLUMNS: Set[str] = set()

    _CATEGORICAL_COLUMNS: Set[str] = {"HomeTeam", "AwayTeam", "FTR"}

    _NUMERIC_COLUMNS: Set[str] = {
        "FTHG",
        "FTAG",
        "HS",
        "AS",
        "HST",
        "AST",
        "HC",
        "AC",
        "HY",
        "AY",
        "HR",
        "AR",
    }

    @classmethod
    def get_required_columns(cls) -> List[str]:
        """
        Get list of all required column names.

        Returns:
            List[str]: List of required column names in order.
        """
        return list(cls._COLUMNS.keys())

    @classmethod
    def get_column_dtype(cls, column_name: str) -> str:
        """
        Get the expected data type for a specific column.

        Args:
            column_name (str): Name of the column.

        Returns:
            str: The expected data type string.

        Raises:
            KeyError: If column_name is not in schema.
        """
        if column_name not in cls._COLUMNS:
            raise KeyError(f"Column '{column_name}' not found in schema")
        return cls._COLUMNS[column_name].value

    @classmethod
    def get_all_dtypes(cls) -> Dict[str, str]:
        """
        Get mapping of all column names to their data types.

        Returns:
            Dict[str, str]: Dictionary mapping column names to data type strings.
        """
        return {col: dtype.value for col, dtype in cls._COLUMNS.items()}

    @classmethod
    def get_categorical_columns(cls) -> Set[str]:
        """
        Get set of categorical column names.

        Returns:
            Set[str]: Set of categorical column names.
        """
        return cls._CATEGORICAL_COLUMNS.copy()

    @classmethod
    def get_numeric_columns(cls) -> Set[str]:
        """
        Get set of numeric column names.

        Returns:
            Set[str]: Set of numeric column names.
        """
        return cls._NUMERIC_COLUMNS.copy()

    @classmethod
    def get_optional_columns(cls) -> Set[str]:
        """
        Get set of optional column names.

        Returns:
            Set[str]: Set of optional column names.
        """
        return cls._OPTIONAL_COLUMNS.copy()

    @classmethod
    def add_column(
        cls,
        column_name: str,
        dtype: DataType,
        optional: bool = False,
        categorical: bool = False,
    ) -> None:
        """
        Add a new column to the schema.

        This method allows extending the schema with new columns for future
        requirements without modifying the base schema definition.

        Args:
            column_name (str): Name of the new column.
            dtype (DataType): Data type of the column.
            optional (bool): Whether the column is optional. Defaults to False.
            categorical (bool): Whether the column is categorical. Defaults to False.

        Raises:
            ValueError: If column_name already exists in schema.
        """
        if column_name in cls._COLUMNS:
            raise ValueError(f"Column '{column_name}' already exists in schema")

        cls._COLUMNS[column_name] = dtype
        if optional:
            cls._OPTIONAL_COLUMNS.add(column_name)
        if categorical:
            cls._CATEGORICAL_COLUMNS.add(column_name)

    @classmethod
    def remove_column(cls, column_name: str) -> None:
        """
        Remove a column from the schema.

        Args:
            column_name (str): Name of the column to remove.

        Raises:
            ValueError: If column_name not found in schema.
        """
        if column_name not in cls._COLUMNS:
            raise ValueError(f"Column '{column_name}' not found in schema")

        del cls._COLUMNS[column_name]
        cls._OPTIONAL_COLUMNS.discard(column_name)
        cls._CATEGORICAL_COLUMNS.discard(column_name)
        cls._NUMERIC_COLUMNS.discard(column_name)

    @classmethod
    def get_schema_info(cls) -> Dict[str, any]:
        """
        Get comprehensive schema information.

        Returns:
            Dict[str, any]: Dictionary containing complete schema metadata.
        """
        return {
            "required_columns": cls.get_required_columns(),
            "dtypes": cls.get_all_dtypes(),
            "categorical_columns": list(cls.get_categorical_columns()),
            "numeric_columns": list(cls.get_numeric_columns()),
            "optional_columns": list(cls.get_optional_columns()),
            "total_columns": len(cls._COLUMNS),
        }
