"""
Persistence logic for the model selection module.
"""

import os
import json
import joblib
from pathlib import Path
from typing import Any, Tuple
from datetime import datetime

from .metadata import ModelSelectionMetadata


def _custom_json_serializer(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def persist_best_model(
    model: Any,
    metadata: ModelSelectionMetadata,
    directory: str = "models"
) -> Tuple[str, str]:
    """
    Saves the best model and its metadata to the specified directory.
    
    Args:
        model: The trained sklearn estimator.
        metadata: The ModelSelectionMetadata instance.
        directory: Directory to save the files.
        
    Returns:
        A tuple containing (model_file_path, metadata_file_path).
    """
    save_dir = Path(directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / "best_model.joblib"
    metadata_path = save_dir / "best_model_metadata.json"
    
    # Save the model
    joblib.dump(model, model_path)
    
    # Save the metadata
    from dataclasses import asdict
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(metadata), f, indent=4, default=_custom_json_serializer)
        
    return str(model_path), str(metadata_path)
