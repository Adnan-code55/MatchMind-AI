"""
Tests for validators in the evaluation module.
"""

import pytest
import numpy as np
from backend.ml.evaluation.validators import (
    validate_arrays,
    validate_probabilities,
    validate_binary_labels
)
from backend.ml.evaluation.exceptions import InvalidInputError

def test_validate_arrays_valid():
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.6])
    # Should not raise
    validate_arrays(y_true, y_pred, y_prob)
    validate_arrays(y_true, y_pred)

def test_validate_arrays_empty():
    with pytest.raises(InvalidInputError, match="must not be empty"):
        validate_arrays(np.array([]), np.array([]))

def test_validate_arrays_length_mismatch():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1, 1])
    with pytest.raises(InvalidInputError, match="Length mismatch"):
        validate_arrays(y_true, y_pred)
        
    y_pred_correct = np.array([0, 1])
    y_prob_wrong = np.array([0.1])
    with pytest.raises(InvalidInputError, match="Length mismatch"):
        validate_arrays(y_true, y_pred_correct, y_prob_wrong)

def test_validate_probabilities_valid():
    y_prob = np.array([0.0, 0.5, 1.0])
    validate_probabilities(y_prob)

def test_validate_probabilities_out_of_bounds():
    with pytest.raises(InvalidInputError, match="in the range"):
        validate_probabilities(np.array([-0.1, 0.5]))
    with pytest.raises(InvalidInputError, match="in the range"):
        validate_probabilities(np.array([0.1, 1.5]))

def test_validate_probabilities_nan():
    with pytest.raises(InvalidInputError, match="NaN"):
        validate_probabilities(np.array([0.5, np.nan]))

def test_validate_binary_labels_valid():
    y_true = np.array([0, 1, 0])
    y_pred = np.array([1, 1, 0])
    validate_binary_labels(y_true, y_pred)

def test_validate_binary_labels_invalid():
    y_true = np.array([0, 2])
    y_pred = np.array([1, 0])
    with pytest.raises(InvalidInputError, match="invalid labels"):
        validate_binary_labels(y_true, y_pred)
        
    y_true2 = np.array([0, 1])
    y_pred2 = np.array([1, -1])
    with pytest.raises(InvalidInputError, match="invalid labels"):
        validate_binary_labels(y_true2, y_pred2)
