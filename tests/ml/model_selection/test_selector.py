"""
Integration tests for the model selector module.
"""

import os
import pytest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from backend.ml.model_selection.selector import ModelSelector

def test_model_selector_integration(tmp_path):
    # Generate dummy data
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
    y_series = pd.Series(y)
    
    X_train, X_val, y_train, y_val = train_test_split(X_df, y_series, test_size=0.2, random_state=42)
    
    # Run selector
    selector = ModelSelector(primary_metric="f1", random_seed=42)
    
    # We will patch persist_best_model to save to tmp_path
    import backend.ml.model_selection.selector as selector_module
    original_persist = selector_module.persist_best_model
    
    def mock_persist(model, metadata, directory="models"):
        return original_persist(model, metadata, directory=str(tmp_path))
        
    selector_module.persist_best_model = mock_persist
    
    try:
        best_model, metadata = selector.run_selection(
            X_train, y_train, X_val, y_val, dataset_name="TestDataset"
        )
        
        # Verify
        assert best_model is not None
        assert metadata.winning_model is not None
        assert metadata.extra["dataset_name"] == "TestDataset"
        assert len(metadata.ranking) > 0
        
        # Verify persistence
        assert os.path.exists(tmp_path / "best_model.joblib")
        assert os.path.exists(tmp_path / "best_model_metadata.json")
    finally:
        # Restore
        selector_module.persist_best_model = original_persist
