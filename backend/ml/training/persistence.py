"""
Persistence for the ML Training module.
"""
import os
import joblib
from typing import Any
from pathlib import Path


def save_model(model: Any, model_name: str, directory: str = "models") -> str:
    """
    Save the trained model to disk using joblib.
    
    Args:
        model: The trained sklearn estimator.
        model_name: Base name for the saved file.
        directory: Directory where the model should be saved.
        
    Returns:
        The path to the saved model file.
    """
    # Create the directory if it doesn't exist
    save_dir = Path(directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped filename to prevent overwriting
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = save_dir / f"{model_name}_{timestamp}.joblib"
    
    joblib.dump(model, file_path)
    return str(file_path)
