"""
Tests for reports generation in the evaluation module.
"""

import json
import csv
from pathlib import Path
from backend.ml.evaluation.reports import export_to_json, export_to_csv
from backend.ml.evaluation.metadata import ModelMetadata, EvaluationMetadata

def test_export_to_json(tmp_path):
    metrics = {"accuracy": 0.9, "f1": 0.85, "confusion_matrix": [[10, 2], [1, 15]]}
    meta = EvaluationMetadata(
        model=ModelMetadata("TestModel", "1.0"),
        dataset_name="TestData",
        num_samples=28,
        split_strategy="Single"
    )
    
    file_path = tmp_path / "report.json"
    export_to_json(metrics, meta, file_path)
    
    assert file_path.exists()
    
    with open(file_path, "r") as f:
        data = json.load(f)
        
    assert data["metrics"]["accuracy"] == 0.9
    assert data["metadata"]["model"]["name"] == "TestModel"

def test_export_to_csv(tmp_path):
    metrics = {"accuracy": 0.9, "f1": 0.85, "confusion_matrix": [[10, 2], [1, 15]]}
    meta = EvaluationMetadata(
        model=ModelMetadata("TestModel", "1.0"),
        dataset_name="TestData",
        num_samples=28,
        split_strategy="Single"
    )
    
    file_path = tmp_path / "report.csv"
    export_to_csv(metrics, meta, file_path)
    
    assert file_path.exists()
    
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 1
    assert rows[0]["model_name"] == "TestModel"
    assert rows[0]["accuracy"] == "0.9"
    # Confusion matrix should not be in CSV since it's a list
    assert "confusion_matrix" not in rows[0]
