"""
Tests for validators in the model selection module.
"""

import pytest
import pandas as pd
from backend.ml.model_selection.validators import (
    validate_models_list,
    validate_selection_dataset
)
from backend.ml.model_selection.exceptions import InvalidModelConfigError

def test_validate_models_list_valid():
    # 'logistic_regression' is in MODEL_REGISTRY
    validate_models_list(["logistic_regression"])

def test_validate_models_list_invalid():
    with pytest.raises(InvalidModelConfigError, match="Unsupported models"):
        validate_models_list(["logistic_regression", "invalid_model"])

def test_validate_models_list_empty():
    with pytest.raises(InvalidModelConfigError, match="must not be empty"):
        validate_models_list([])

def test_validate_selection_dataset_valid():
    X_train = pd.DataFrame({"a": [1, 2]})
    y_train = pd.Series([0, 1])
    X_val = pd.DataFrame({"a": [3]})
    y_val = pd.Series([0])
    
    validate_selection_dataset(X_train, y_train, X_val, y_val)

def test_validate_selection_dataset_empty():
    X_train = pd.DataFrame()
    y_train = pd.Series()
    X_val = pd.DataFrame({"a": [3]})
    y_val = pd.Series([0])
    
    with pytest.raises(ValueError, match="must not be empty"):
        validate_selection_dataset(X_train, y_train, X_val, y_val)

def test_validate_selection_dataset_mismatch():
    X_train = pd.DataFrame({"a": [1, 2]})
    y_train = pd.Series([0])
    X_val = pd.DataFrame({"a": [3]})
    y_val = pd.Series([0])
    
    with pytest.raises(ValueError, match="same length"):
        validate_selection_dataset(X_train, y_train, X_val, y_val)
