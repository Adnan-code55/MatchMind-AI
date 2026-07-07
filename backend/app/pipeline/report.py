"""
Pipeline report generation for MatchMind AI.

This module defines the structure of execution reports and provides JSON
serialization for auditing and observability.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class PipelineReport:
    """Execution report for a single pipeline run."""

    execution_id: str
    start_time: str
    finish_time: str
    duration_seconds: float
    rows_loaded: int
    rows_removed: int
    duplicate_rows: int
    missing_values_fixed: int
    validation_errors: List[str]
    warnings: List[str]
    output_file: str
    metadata_file: str
    report_file: str
    pipeline_version: str
    status: str
    league: str
    season: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the pipeline report to a serializable dictionary."""
        return asdict(self)

    def save(self, directory: Path) -> Path:
        """Save the pipeline report as a JSON file."""
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / f"pipeline_report_{self.execution_id}.json"
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(self.to_dict(), report_file, indent=2)
        return report_path
