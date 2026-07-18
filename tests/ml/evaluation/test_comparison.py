"""
Tests for model comparison in the evaluation module.
"""

import pytest
from backend.ml.evaluation.comparison import ModelComparator
from backend.ml.evaluation.exceptions import MissingMetricError

def test_rank_models_f1():
    comparator = ModelComparator(primary_metric="f1", higher_is_better=True)
    results = [
        {"model_name": "ModelA", "metrics": {"f1": 0.8}},
        {"model_name": "ModelB", "metrics": {"f1": 0.9}},
        {"model_name": "ModelC", "metrics": {"f1": 0.85}},
    ]
    
    ranked = comparator.rank_models(results)
    
    assert ranked[0]["model_name"] == "ModelB"
    assert ranked[1]["model_name"] == "ModelC"
    assert ranked[2]["model_name"] == "ModelA"

def test_rank_models_log_loss():
    comparator = ModelComparator(primary_metric="log_loss", higher_is_better=False)
    results = [
        {"model_name": "ModelA", "metrics": {"log_loss": 0.5}},
        {"model_name": "ModelB", "metrics": {"log_loss": 0.3}},
        {"model_name": "ModelC", "metrics": {"log_loss": 0.4}},
    ]
    
    ranked = comparator.rank_models(results)
    
    assert ranked[0]["model_name"] == "ModelB"
    assert ranked[1]["model_name"] == "ModelC"
    assert ranked[2]["model_name"] == "ModelA"

def test_rank_models_missing_metric():
    comparator = ModelComparator(primary_metric="f1")
    results = [
        {"model_name": "ModelA", "metrics": {"f1": 0.8}},
        {"model_name": "ModelB", "metrics": {"accuracy": 0.9}},
    ]
    
    with pytest.raises(MissingMetricError, match="missing the primary metric 'f1'"):
        comparator.rank_models(results)
