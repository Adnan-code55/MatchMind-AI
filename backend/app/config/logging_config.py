"""
Logging configuration for MatchMind AI pipeline.

This module centralizes execution log file creation and ensures that each
pipeline run creates a dedicated log file with execution context.
"""

import logging
from pathlib import Path

from .settings import PipelineSettings
from ..data.logger import PipelineLogger


def get_log_file_path(settings: PipelineSettings, execution_id: str) -> Path:
    """Create and return the path for the current execution log file."""
    log_file = settings.log_path / f"pipeline_{execution_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file


def configure_logging(
    settings: PipelineSettings,
    execution_id: str,
    level: int = logging.INFO,
) -> Path:
    """Configure pipeline logging for the current execution."""
    log_file = get_log_file_path(settings, execution_id)
    PipelineLogger.initialize(
        log_file=str(log_file),
        level=level,
        execution_id=execution_id,
    )
    return log_file
