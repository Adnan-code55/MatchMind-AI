"""
Validators for the model selection module.
"""

from typing import List
import pandas as pd

from backend.ml.training.models import get_supported_models
from .exceptions import InvalidModelConfigError


def validate_models_list(models_to_train: List[str]) -> None:
    """
    Validates that the provided models are supported.
    
    Args:
        models_to_train: List of model names to evaluate.
        
    Raises:
        InvalidModelConfigError: If the list is empty or contains unsupported models.
    """
    if not models_to_train:
        raise InvalidModelConfigError("The list of models to train must not be empty.")
        
    supported = set(get_supported_models())
    invalid_models = set(models_to_train) - supported
    
    if invalid_models:
        raise InvalidModelConfigError(
            f"Unsupported models provided: {invalid_models}. "
            f"Supported models are: {supported}"
        )


def validate_selection_dataset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series
) -> None:
    """
    Validates the dataset provided for model selection.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_validation: Validation features.
        y_validation: Validation targets.
        
    Raises:
        ValueError: If datasets are empty or shapes mismatch.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("Training dataset must not be empty.")
    if X_validation.empty or y_validation.empty:
        raise ValueError("Validation dataset must not be empty.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")
    if len(X_validation) != len(y_validation):
        raise ValueError("X_validation and y_validation must have the same length.")
