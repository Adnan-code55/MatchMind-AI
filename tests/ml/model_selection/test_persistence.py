"""
Tests for persistence in the model selection module.
"""

import os
import json
from sklearn.linear_model import LogisticRegression
from backend.ml.model_selection.persistence import persist_best_model
from backend.ml.model_selection.metadata import ModelSelectionMetadata

def test_persist_best_model(tmp_path):
    model = LogisticRegression()
    metadata = ModelSelectionMetadata(
        winning_model="LogisticRegression",
        evaluation_metrics={"f1": 0.8},
        ranking=[{"model_name": "LogisticRegression", "metrics": {"f1": 0.8}}]
    )
    
    save_dir = tmp_path / "models"
    model_path, metadata_path = persist_best_model(model, metadata, directory=str(save_dir))
    
    assert os.path.exists(model_path)
    assert os.path.exists(metadata_path)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    assert data["winning_model"] == "LogisticRegression"
    assert "timestamp" in data
