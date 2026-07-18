"""
Tests for EvaluationEngine in the evaluation module.
"""

import pytest
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from backend.ml.evaluation.evaluator import EvaluationEngine
from backend.ml.evaluation.metadata import ModelMetadata


class DummyClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self
        
    def predict(self, X):
        return np.zeros(len(X), dtype=int)
        
    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 1.0  # Always predict class 0 with 1.0 probability
        return probs


def test_evaluate_predictions():
    engine = EvaluationEngine()
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4])
    
    meta = ModelMetadata("Dummy", "1.0")
    metrics, eval_meta = engine.evaluate_predictions(
        y_true, y_pred, y_prob, meta, "TestData"
    )
    
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert eval_meta.model.name == "Dummy"
    assert eval_meta.dataset_name == "TestData"
    assert eval_meta.num_samples == 4


def test_cross_validate_kfold():
    engine = EvaluationEngine()
    model = DummyClassifier()
    
    X = np.random.rand(20, 5)
    y = np.array([0]*10 + [1]*10)
    
    metrics, eval_meta = engine.cross_validate(
        model, X, y, n_splits=2, stratified=False, random_state=42
    )
    
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert eval_meta.model.name == "DummyClassifier"
    assert eval_meta.split_strategy == "2-fold KFold"
    assert eval_meta.num_samples == 20


def test_cross_validate_stratified():
    engine = EvaluationEngine()
    model = DummyClassifier()
    
    X = np.random.rand(20, 5)
    y = np.array([0]*10 + [1]*10)
    
    metrics, eval_meta = engine.cross_validate(
        model, X, y, n_splits=2, stratified=True, random_state=42
    )
    
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert eval_meta.model.name == "DummyClassifier"
    assert eval_meta.split_strategy == "2-fold StratifiedKFold"
    assert eval_meta.num_samples == 20
