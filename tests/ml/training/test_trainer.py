import os
import pytest
import pandas as pd
from pathlib import Path
from backend.ml.training import (
    ModelTrainer,
    UnsupportedModelError,
    EmptyDatasetError,
    MismatchedDimensionsError,
    InvalidInputTypeError,
)

@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [5, 6, 7, 8]})
    y_train = pd.Series([0, 1, 0, 1])
    X_val = pd.DataFrame({"feat1": [1, 2], "feat2": [5, 6]})
    y_val = pd.Series([0, 1])
    return X_train, y_train, X_val, y_val

def test_train_valid_model(sample_data, tmp_path):
    X_train, y_train, X_val, y_val = sample_data
    
    # Mocking save directory for tests
    import backend.ml.training.trainer as trainer_mod
    import backend.ml.training.persistence as persistence
    original_save = persistence.save_model
    
    def mock_save_model(model, name, directory="models"):
        return original_save(model, name, directory=str(tmp_path))
    
    trainer_mod.save_model = mock_save_model
    
    trainer = ModelTrainer(random_seed=42)
    trainer.train(
        model_name="logistic_regression",
        X_train=X_train,
        y_train=y_train,
        X_validation=X_val,
        y_validation=y_val
    )
    
    # Assert model exists
    assert trainer.model is not None
    
    # Assert metadata
    meta = trainer.get_metadata()
    assert meta is not None
    assert meta.algorithm == "logistic_regression"
    assert meta.dataset_size == 4
    assert meta.feature_count == 2
    assert meta.random_seed == 42
    
    # Assert file saved
    saved_files = list(tmp_path.glob("*.joblib"))
    assert len(saved_files) == 1
    
    # Restore
    trainer_mod.save_model = original_save

def test_unsupported_model(sample_data):
    X_train, y_train, X_val, y_val = sample_data
    trainer = ModelTrainer()
    with pytest.raises(UnsupportedModelError):
        trainer.train(
            model_name="unsupported_model",
            X_train=X_train,
            y_train=y_train,
            X_validation=X_val,
            y_validation=y_val
        )

def test_empty_dataset(sample_data):
    X_train, y_train, X_val, y_val = sample_data
    trainer = ModelTrainer()
    with pytest.raises(EmptyDatasetError):
        trainer.train(
            model_name="logistic_regression",
            X_train=pd.DataFrame(),
            y_train=pd.Series(),
            X_validation=X_val,
            y_validation=y_val
        )

def test_mismatched_dimensions(sample_data):
    X_train, y_train, X_val, y_val = sample_data
    trainer = ModelTrainer()
    with pytest.raises(MismatchedDimensionsError):
        trainer.train(
            model_name="logistic_regression",
            X_train=X_train,
            y_train=y_train.iloc[:2], # Mismatched target size
            X_validation=X_val,
            y_validation=y_val
        )

def test_invalid_input_type(sample_data):
    X_train, y_train, X_val, y_val = sample_data
    trainer = ModelTrainer()
    with pytest.raises(InvalidInputTypeError):
        trainer.train(
            model_name="logistic_regression",
            X_train=[[1, 2], [3, 4]], # Not a DataFrame
            y_train=y_train,
            X_validation=X_val,
            y_validation=y_val
        )
