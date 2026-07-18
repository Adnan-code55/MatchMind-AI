import pytest
import os
from pathlib import Path
from backend.ml.prediction.loaders import ModelLoader
from backend.ml.prediction.exceptions import ModelLoadError

def test_model_loader_directory_not_found():
    loader = ModelLoader(directory="non_existent_dir_12345")
    with pytest.raises(ModelLoadError, match="Directory not found"):
        loader.load_best_model()

def test_model_loader_no_models(tmp_path):
    loader = ModelLoader(directory=str(tmp_path))
    with pytest.raises(ModelLoadError, match="No models found"):
        loader.load_best_model()

from unittest.mock import patch

def test_model_loader_load_best_model_file(tmp_path):
    with patch("backend.ml.prediction.loaders.joblib.load") as mock_load:
        mock_load.return_value = "mocked_model"

        # Create dummy best_model.joblib
        best_model_file = tmp_path / "best_model.joblib"
        best_model_file.touch()

        loader = ModelLoader(directory=str(tmp_path))
        model = loader.load_best_model()

        assert model == "mocked_model"
        mock_load.assert_called_once_with(best_model_file)

def test_model_loader_fallback_latest_model(tmp_path):
    with patch("backend.ml.prediction.loaders.joblib.load") as mock_load:
        mock_load.return_value = "mocked_fallback_model"

        # Create dummy models with different timestamps
        old_model = tmp_path / "model1.joblib"
        new_model = tmp_path / "model2.joblib"
        
        old_model.touch()
        os.utime(old_model, (1000, 1000))
        
        new_model.touch()
        os.utime(new_model, (2000, 2000))

        loader = ModelLoader(directory=str(tmp_path))
        model = loader.load_best_model()

        assert model == "mocked_fallback_model"
        mock_load.assert_called_once_with(new_model)
