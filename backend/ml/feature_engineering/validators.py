"""
Validation utilities for the Feature Engineering Engine.

This module provides functions to validate datasets before generating
historical features.
"""

from typing import List
import pandas as pd

from .exceptions import MissingColumnError


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate that the dataset contains all required columns.

    Args:
        df: The pandas DataFrame to validate.
        required_columns: A list of column names that must be present.

    Raises:
        MissingColumnError: If any of the required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise MissingColumnError(
            f"Dataset is missing required columns for feature engineering: {', '.join(missing)}"
        )
