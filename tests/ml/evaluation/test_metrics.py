"""
Tests for metrics computation in the evaluation module.
"""

import pytest
import numpy as np
from backend.ml.evaluation.metrics import compute_classification_metrics
from backend.ml.evaluation.exceptions import MetricComputationError, InvalidInputError

def test_compute_classification_metrics_no_prob():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    metrics = compute_classification_metrics(y_true, y_pred)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert "roc_auc" not in metrics
    assert "log_loss" not in metrics

def test_compute_classification_metrics_with_prob():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    
    assert "roc_auc" in metrics
    assert "log_loss" in metrics

def test_compute_classification_metrics_invalid_input():
    # Length mismatch
    with pytest.raises(MetricComputationError):
        compute_classification_metrics(np.array([0, 1]), np.array([0]))
        
    # Invalid probs
    with pytest.raises(MetricComputationError):
        compute_classification_metrics(np.array([0, 1]), np.array([0, 1]), np.array([1.5, 0.5]))
