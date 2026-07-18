"""
Tests for metadata classes in the model selection module.
"""

from datetime import datetime
from backend.ml.model_selection.metadata import ModelSelectionMetadata

def test_model_selection_metadata():
    meta = ModelSelectionMetadata(
        winning_model="BestModel",
        evaluation_metrics={"f1": 0.95},
        ranking=[{"model_name": "BestModel", "metrics": {"f1": 0.95}}]
    )
    
    assert meta.winning_model == "BestModel"
    assert meta.evaluation_metrics["f1"] == 0.95
    assert len(meta.ranking) == 1
    assert isinstance(meta.timestamp, datetime)
    assert meta.extra == {}
