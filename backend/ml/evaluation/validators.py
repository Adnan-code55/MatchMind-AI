"""
Validators for the evaluation module.

Provides utility functions to validate inputs before evaluation.
"""

from typing import Optional
import numpy as np

from .exceptions import InvalidInputError


def validate_arrays(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> None:
    """
    Validates that y_true, y_pred, and optionally y_prob have consistent lengths and shapes.
    
    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        y_prob: Optional array of prediction probabilities.
        
    Raises:
        InvalidInputError: If shapes are inconsistent or empty.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise InvalidInputError("Arrays y_true and y_pred must not be empty.")
        
    if len(y_true) != len(y_pred):
        raise InvalidInputError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)}).")
        
    if y_prob is not None:
        if len(y_true) != len(y_prob):
            raise InvalidInputError(f"Length mismatch: y_true ({len(y_true)}) vs y_prob ({len(y_prob)}).")


def validate_probabilities(y_prob: np.ndarray) -> None:
    """
    Validates that probabilities are within the [0, 1] range.
    
    Args:
        y_prob: Array of prediction probabilities.
        
    Raises:
        InvalidInputError: If any value is outside [0, 1] or is NaN.
    """
    if np.isnan(y_prob).any():
        raise InvalidInputError("Probabilities contain NaN values.")
        
    if (y_prob < 0.0).any() or (y_prob > 1.0).any():
        raise InvalidInputError("Probabilities must be in the range [0, 1].")


def validate_binary_labels(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Validates that labels are binary (0 or 1).
    
    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        
    Raises:
        InvalidInputError: If labels contain values other than 0 and 1.
    """
    valid_labels = {0, 1}
    unique_true = set(np.unique(y_true))
    unique_pred = set(np.unique(y_pred))
    
    if not unique_true.issubset(valid_labels):
        raise InvalidInputError(f"y_true contains invalid labels: {unique_true - valid_labels}. Expected binary labels.")
        
    if not unique_pred.issubset(valid_labels):
        raise InvalidInputError(f"y_pred contains invalid labels: {unique_pred - valid_labels}. Expected binary labels.")
