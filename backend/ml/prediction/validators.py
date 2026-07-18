"""
Validators for the Prediction module.
"""

import pandas as pd
from .exceptions import InvalidInputDataError


def validate_prediction_input(df: pd.DataFrame) -> None:
    """
    Validates the input DataFrame for prediction.
    
    Args:
        df: The pandas DataFrame to validate.
        
    Raises:
        InvalidInputDataError: If the input is not a DataFrame or is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise InvalidInputDataError("Input data must be a pandas DataFrame.")
    
    if df.empty:
        raise InvalidInputDataError("Input data cannot be empty.")
