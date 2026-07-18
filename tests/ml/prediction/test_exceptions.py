import pytest
from backend.ml.prediction.exceptions import (
    PredictionError,
    ModelLoadError,
    InvalidInputDataError,
    FeatureMismatchError
)

def test_exceptions_inheritance():
    assert issubclass(ModelLoadError, PredictionError)
    assert issubclass(InvalidInputDataError, PredictionError)
    assert issubclass(FeatureMismatchError, PredictionError)
