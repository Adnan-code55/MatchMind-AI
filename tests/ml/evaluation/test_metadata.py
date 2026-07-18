"""
Tests for metadata classes in the evaluation module.
"""

from backend.ml.evaluation.metadata import ModelMetadata, EvaluationMetadata
from datetime import datetime

def test_model_metadata():
    model = ModelMetadata(name="RandomForest", version="v1.0", hyperparameters={"n_estimators": 100})
    assert model.name == "RandomForest"
    assert model.version == "v1.0"
    assert model.hyperparameters["n_estimators"] == 100

def test_evaluation_metadata():
    model = ModelMetadata(name="TestModel", version="1.0")
    meta = EvaluationMetadata(
        model=model,
        dataset_name="TestDataset",
        num_samples=1000,
        split_strategy="KFold"
    )
    
    assert meta.model.name == "TestModel"
    assert meta.dataset_name == "TestDataset"
    assert meta.num_samples == 1000
    assert meta.split_strategy == "KFold"
    assert isinstance(meta.timestamp, datetime)
    assert meta.extra == {}
