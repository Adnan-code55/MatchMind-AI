"""
Tests for the command line interface.

This module verifies that the pipeline script parses arguments and returns the
expected exit status.
"""

import os
import runpy
import sys
from pathlib import Path

import pytest

from backend.app.config.settings import PipelineSettings


def test_cli_help_message(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_RAW_PATH", str(tmp_path / "raw"))
    monkeypatch.setenv("DATA_PROCESSED_PATH", str(tmp_path / "processed"))
    monkeypatch.setenv("DATA_METADATA_PATH", str(tmp_path / "metadata"))
    monkeypatch.setenv("REPORT_PATH", str(tmp_path / "reports"))
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "logs"))

    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(Path("scripts/run_pipeline.py"), run_name="__main__")

    assert excinfo.value.code == 0


def test_cli_parses_arguments(monkeypatch, tmp_path):
    raw_path = tmp_path / "input"
    processed_path = tmp_path / "output"
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
    raw_path.mkdir(parents=True, exist_ok=True)
    (raw_path / "matches.csv").write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HY,AY,HR,AR\n"
        "2023-01-01,Arsenal,Chelsea,2,1,H,10,6,5,2,8,5,2,1,0,0\n"
    )

    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--league", "EPL", "--season", "2024", "--source", str(raw_path), "--output", str(processed_path)])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(Path("scripts/run_pipeline.py"), run_name="__main__")

    assert excinfo.value.code == 0
    assert list(report_path.glob("pipeline_report_*.json"))
