"""
Reports generation for the evaluation module.

Handles serialization and exporting of evaluation results to JSON and CSV formats.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime

from .metadata import EvaluationMetadata


def _custom_json_serializer(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def export_to_json(
    metrics: Dict[str, Any],
    metadata: EvaluationMetadata,
    filepath: Union[str, Path]
) -> None:
    """
    Exports evaluation metrics and metadata to a JSON file.
    
    Args:
        metrics: Dictionary of computed metrics.
        metadata: Evaluation metadata.
        filepath: Destination file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    from dataclasses import asdict
    
    report = {
        "metadata": asdict(metadata),
        "metrics": metrics
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, default=_custom_json_serializer)


def export_to_csv(
    metrics: Dict[str, Any],
    metadata: EvaluationMetadata,
    filepath: Union[str, Path]
) -> None:
    """
    Exports evaluation metrics and metadata to a flattened CSV file.
    Note: Complex types like confusion matrix are not exported to CSV or are stringified.
    
    Args:
        metrics: Dictionary of computed metrics.
        metadata: Evaluation metadata.
        filepath: Destination file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    row = {
        "model_name": metadata.model.name,
        "model_version": metadata.model.version,
        "dataset_name": metadata.dataset_name,
        "num_samples": metadata.num_samples,
        "split_strategy": metadata.split_strategy,
        "timestamp": metadata.timestamp.isoformat()
    }
    
    # Add flat metrics
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool)):
            row[k] = v
            
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
