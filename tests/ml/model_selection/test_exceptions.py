"""
Tests for exceptions in the model selection module.
"""

import pytest
from backend.ml.model_selection.exceptions import (
    ModelSelectionError,
    NoModelsTrainedError,
    InvalidModelConfigError
)

def test_model_selection_error():
    with pytest.raises(ModelSelectionError):
        raise ModelSelectionError("base error")

def test_no_models_trained_error():
    with pytest.raises(NoModelsTrainedError):
        raise NoModelsTrainedError("no models")
        
def test_invalid_model_config_error():
    with pytest.raises(InvalidModelConfigError):
        raise InvalidModelConfigError("invalid config")
