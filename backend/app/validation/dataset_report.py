"""
Validation report definitions for engineered datasets.

This module defines the structure and serialization of feature validation reports
that can be used to inspect dataset quality before model training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetValidationReport:
    """Structured report for dataset validation results."""

    rows: int
    columns: int
    feature_count: int
    target_column: Optional[str]
    target_distribution: Dict[str, Any]
    missing_values: Dict[str, int]
    infinite_values: Dict[str, int]
    duplicate_rows: int
    duplicate_columns: List[str]
    constant_features: List[str]
    near_constant_features: Dict[str, float]
    invalid_feature_types: Dict[str, str]
    correlation_matrix: Dict[str, Dict[str, float]]
    highly_correlated_features: List[Dict[str, Any]]
    numeric_statistics: Dict[str, Dict[str, Any]]
    feature_summary: Dict[str, Dict[str, Any]]
    warnings: List[str]
    recommendations: List[str]
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation report to a serializable dictionary."""
        return asdict(self)

    def save(self, path: Path) -> Path:
        """Save the validation report as a JSON file."""
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as report_file:
            json.dump(self.to_dict(), report_file, indent=2)
        return path
