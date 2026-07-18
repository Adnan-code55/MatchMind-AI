"""
Validators for the ML Training module.
"""
from typing import Any
import pandas as pd
from .exceptions import (
    UnsupportedModelError,
    EmptyDatasetError,
    MismatchedDimensionsError,
    InvalidInputTypeError,
)


def validate_model_name(model_name: str, supported_models: list[str]) -> None:
    """Validate that the given model name is supported."""
    if model_name not in supported_models:
        raise UnsupportedModelError(
            f"Model '{model_name}' is not supported. Supported models: {supported_models}"
        )


def validate_dataset(X: Any, y: Any, dataset_name: str = "dataset") -> None:
    """Validate dataset types, dimensions, and emptiness."""
    if not isinstance(X, pd.DataFrame):
        raise InvalidInputTypeError(f"X_{dataset_name} must be a pandas DataFrame.")
    
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise InvalidInputTypeError(f"y_{dataset_name} must be a pandas Series or DataFrame.")

    if X.empty:
        raise EmptyDatasetError(f"X_{dataset_name} cannot be empty.")
        
    if y.empty:
        raise EmptyDatasetError(f"y_{dataset_name} cannot be empty.")

    if len(X) != len(y):
        raise MismatchedDimensionsError(
            f"Mismatched dimensions in {dataset_name}: X has {len(X)} rows, y has {len(y)} rows."
        )
