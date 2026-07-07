"""
Pipeline orchestrator for MatchMind AI.

This module provides a thin orchestration layer to construct and execute the
business pipeline using centralized configuration.
"""

from pathlib import Path
from typing import Optional

from ..config.settings import PipelineSettings
from ..data.logger import PipelineLogger
from .pipeline import Pipeline


class PipelineOrchestrator:
    """High-level orchestrator for pipeline execution."""

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        """Initialize the orchestrator with settings."""
        self.settings = settings or PipelineSettings.load()
        PipelineLogger.log_info(__name__, "PipelineOrchestrator initialized")

    def run(
        self,
        league: Optional[str] = None,
        season: Optional[str] = None,
        source: Optional[str] = None,
        output: Optional[str] = None,
    ) -> PipelineReport:
        """Execute the pipeline with optional runtime overrides."""
        source_path = Path(source) if source else self.settings.data_raw_path
        output_path = Path(output) if output else self.settings.data_processed_path

        pipeline = Pipeline(
            settings=self.settings,
            league=league,
            season=season,
            source_path=source_path,
            output_path=output_path,
        )
        return pipeline.run()
