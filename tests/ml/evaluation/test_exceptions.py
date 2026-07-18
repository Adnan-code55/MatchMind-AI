"""
Tests for exceptions in the evaluation module.
"""

import pytest
from backend.ml.evaluation.exceptions import (
    EvaluationError,
    InvalidInputError,
    MetricComputationError,
    MissingMetricError
)

def test_evaluation_error():
    with pytest.raises(EvaluationError):
        raise EvaluationError("base error")

def test_invalid_input_error():
    with pytest.raises(InvalidInputError):
        raise InvalidInputError("invalid input")
        
def test_metric_computation_error():
    with pytest.raises(MetricComputationError):
        raise MetricComputationError("metric failed")

def test_missing_metric_error():
    with pytest.raises(MissingMetricError):
        raise MissingMetricError("metric missing")
