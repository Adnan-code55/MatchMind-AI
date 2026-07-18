"""
Model loader for the Prediction module.
"""

import os
import joblib
from pathlib import Path
from typing import Any

from .exceptions import ModelLoadError


class ModelLoader:
    """Class responsible for loading the trained model for prediction."""

    def __init__(self, directory: str = "models") -> None:
        """
        Initializes the ModelLoader.
        
        Args:
            directory: Directory where the models are saved.
        """
        self.directory = Path(directory)

    def load_best_model(self) -> Any:
        """
        Loads the best saved model. It prioritizes 'best_model.joblib',
        and falls back to the most recently saved model if that doesn't exist.
        
        Returns:
            The loaded sklearn estimator.
            
        Raises:
            ModelLoadError: If no model can be found or loaded.
        """
        if not self.directory.exists() or not self.directory.is_dir():
            raise ModelLoadError(f"Directory not found: {self.directory}")

        best_model_path = self.directory / "best_model.joblib"
        
        if best_model_path.exists():
            try:
                return joblib.load(best_model_path)
            except Exception as e:
                raise ModelLoadError(f"Failed to load best_model.joblib: {e}")

        # Fallback to the most recent model
        joblib_files = list(self.directory.glob("*.joblib"))
        if not joblib_files:
            raise ModelLoadError(f"No models found in {self.directory}")

        # Sort by modification time
        latest_model_path = max(joblib_files, key=os.path.getmtime)
        try:
            return joblib.load(latest_model_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to load fallback model {latest_model_path.name}: {e}")
