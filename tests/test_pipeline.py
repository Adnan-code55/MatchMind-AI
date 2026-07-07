"""
Tests for pipeline orchestration, report generation, and metadata versioning.

These tests validate that the pipeline executes end-to-end and saves artifacts
in the configured output locations.
"""

import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from backend.app.config.settings import PipelineSettings
from backend.app.data.exceptions import DataValidationError
from backend.app.pipeline.orchestrator import PipelineOrchestrator


def create_sample_match_file(directory: Path) -> Path:
    """Create a valid sample match CSV file in the directory."""
    data = {
        "Date": ["2023-01-01", "2023-01-02"],
        "HomeTeam": ["Arsenal", "Liverpool"],
        "AwayTeam": ["Chelsea", "Tottenham"],
        "FTHG": ["2", "1"],
        "FTAG": ["1", "1"],
        "FTR": ["H", "D"],
        "HS": ["10", "12"],
        "AS": ["7", "8"],
        "HST": ["5", "4"],
        "AST": ["3", "4"],
        "HC": ["8", "6"],
        "AC": ["3", "5"],
        "HY": ["1", "2"],
        "AY": ["2", "1"],
        "HR": ["0", "0"],
        "AR": ["0", "1"],
    }
    csv_path = directory / "matches.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


def test_pipeline_executes_and_writes_artifacts(monkeypatch, tmp_path):
    """Pipeline should execute successfully and save report, metadata, and output."""
    raw_path = tmp_path / "data" / "raw"
    processed_path = tmp_path / "data" / "processed"
    metadata_path = tmp_path / "metadata"
    report_path = tmp_path / "reports"
    log_path = tmp_path / "logs"

    monkeypatch.setenv("DATA_RAW_PATH", str(raw_path))
    monkeypatch.setenv("DATA_PROCESSED_PATH", str(processed_path))
    monkeypatch.setenv("DATA_METADATA_PATH", str(metadata_path))
    monkeypatch.setenv("REPORT_PATH", str(report_path))
    monkeypatch.setenv("LOG_PATH", str(log_path))
    monkeypatch.setenv("DEFAULT_LEAGUE", "EPL")
    monkeypatch.setenv("PIPELINE_VERSION", "1.1.0")

    settings = PipelineSettings.load()
    create_sample_match_file(settings.data_raw_path)

    orchestrator = PipelineOrchestrator(settings)
    report = orchestrator.run(league="EPL", season="2024")

    assert report.status == "success"
    assert report.output_file
    assert report.metadata_file
    assert Path(report.output_file).exists()
    assert Path(report.metadata_file).exists()
    assert Path(report.report_file).exists()

    with Path(report.metadata_file).open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    assert metadata["league"] == "EPL"
    assert metadata["season"] == "2024"
    assert metadata["pipeline_version"] == "1.1.0"
    assert metadata["rows"] == 2


def test_pipeline_stops_when_validation_fails(monkeypatch, tmp_path):
    """Pipeline should stop safely without saving output when validation fails."""
    raw_path = tmp_path / "data" / "raw"
    processed_path = tmp_path / "data" / "processed"
    metadata_path = tmp_path / "metadata"
    report_path = tmp_path / "reports"
    log_path = tmp_path / "logs"

    monkeypatch.setenv("DATA_RAW_PATH", str(raw_path))
    monkeypatch.setenv("DATA_PROCESSED_PATH", str(processed_path))
    monkeypatch.setenv("DATA_METADATA_PATH", str(metadata_path))
    monkeypatch.setenv("REPORT_PATH", str(report_path))
    monkeypatch.setenv("LOG_PATH", str(log_path))
    monkeypatch.setenv("DEFAULT_LEAGUE", "EPL")

    settings = PipelineSettings.load()
    invalid_data = {
        "Date": ["2023-01-01", "2099-01-01"],
        "HomeTeam": ["Arsenal", "Liverpool"],
        "AwayTeam": ["Chelsea", "Tottenham"],
        "FTHG": ["2", "1"],
        "FTAG": ["1", "1"],
        "FTR": ["H", "D"],
        "HS": ["10", "12"],
        "AS": ["7", "8"],
        "HST": ["5", "4"],
        "AST": ["3", "4"],
        "HC": ["8", "6"],
        "AC": ["3", "5"],
        "HY": ["1", "2"],
        "AY": ["2", "1"],
        "HR": ["0", "0"],
        "AR": ["0", "1"],
    }
    invalid_path = settings.data_raw_path / "invalid_matches.csv"
    pd.DataFrame(invalid_data).to_csv(invalid_path, index=False)

    orchestrator = PipelineOrchestrator(settings)

    with pytest.raises(DataValidationError):
        orchestrator.run(league="EPL", season="2024")

    assert not list(settings.data_processed_path.glob("*.csv"))
    assert any(settings.report_path.glob("pipeline_report_*.json"))
