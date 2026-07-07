"""
Tests for centralized pipeline configuration.

This module validates that environment-aware settings load correctly and that
configured directories are created as expected.
"""

import os
from pathlib import Path
import tempfile

import pytest

from backend.app.config.settings import PipelineSettings


def test_settings_loads_defaults(monkeypatch, tmp_path):
    """PipelineSettings.load should resolve default paths and create directories."""
    monkeypatch.delenv("DATA_RAW_PATH", raising=False)
    monkeypatch.delenv("DATA_PROCESSED_PATH", raising=False)
    monkeypatch.delenv("DATA_METADATA_PATH", raising=False)
    monkeypatch.delenv("REPORT_PATH", raising=False)
    monkeypatch.delenv("LOG_PATH", raising=False)
    monkeypatch.delenv("DEFAULT_LEAGUE", raising=False)
    monkeypatch.delenv("PIPELINE_VERSION", raising=False)
    monkeypatch.delenv("RANDOM_SEED", raising=False)

    monkeypatch.chdir(tmp_path)

    settings = PipelineSettings.load()

    assert settings.data_raw_path.exists()
    assert settings.data_processed_path.exists()
    assert settings.data_metadata_path.exists()
    assert settings.report_path.exists()
    assert settings.log_path.exists()
    assert settings.default_league == "EPL"
    assert settings.pipeline_version == "1.1.0"
    assert isinstance(settings.random_seed, int)


def test_settings_respects_environment(monkeypatch, tmp_path):
    """PipelineSettings.load should use environment overrides when provided."""
    raw_path = tmp_path / "raw"
    processed_path = tmp_path / "processed"
    metadata_path = tmp_path / "metadata"
    report_path = tmp_path / "reports"
    log_path = tmp_path / "logs"

    monkeypatch.setenv("DATA_RAW_PATH", str(raw_path))
    monkeypatch.setenv("DATA_PROCESSED_PATH", str(processed_path))
    monkeypatch.setenv("DATA_METADATA_PATH", str(metadata_path))
    monkeypatch.setenv("REPORT_PATH", str(report_path))
    monkeypatch.setenv("LOG_PATH", str(log_path))
    monkeypatch.setenv("DEFAULT_LEAGUE", "LA_LIGA")
    monkeypatch.setenv("PIPELINE_VERSION", "1.1.1")
    monkeypatch.setenv("RANDOM_SEED", "123")

    settings = PipelineSettings.load()

    assert settings.data_raw_path == raw_path.resolve()
    assert settings.data_processed_path == processed_path.resolve()
    assert settings.data_metadata_path == metadata_path.resolve()
    assert settings.report_path == report_path.resolve()
    assert settings.log_path == log_path.resolve()
    assert settings.default_league == "LA_LIGA"
    assert settings.pipeline_version == "1.1.1"
    assert settings.random_seed == 123
    assert raw_path.exists()
    assert processed_path.exists()
    assert metadata_path.exists()
    assert report_path.exists()
    assert log_path.exists()
