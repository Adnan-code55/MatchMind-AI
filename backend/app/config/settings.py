"""
Pipeline settings for MatchMind AI.

This module loads environment-aware settings and exposes a stable configuration
object used by pipeline orchestration, file management, and logging.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .constants import DEFAULT_LEAGUE, PIPELINE_VERSION, RANDOM_SEED, SUPPORTED_LEAGUES
from .paths import (
    ensure_directories,
    get_default_log_path,
    get_default_metadata_path,
    get_default_processed_path,
    get_default_raw_path,
    get_default_report_path,
    resolve_path,
)


@dataclass(frozen=True)
class PipelineSettings:
    """
    Configuration settings for the pipeline.

    Attributes:
        data_raw_path: Path to raw data directory.
        data_processed_path: Path to processed output directory.
        data_metadata_path: Path to dataset metadata directory.
        report_path: Path to pipeline report directory.
        log_path: Path to execution log directory.
        default_league: Default league code used when no league is specified.
        supported_leagues: Supported league metadata dictionary.
        pipeline_version: Version of the pipeline.
        random_seed: Random seed for reproducibility.
    """

    data_raw_path: Path
    data_processed_path: Path
    data_metadata_path: Path
    report_path: Path
    log_path: Path
    default_league: str
    supported_leagues: Dict[str, Dict[str, str]]
    pipeline_version: str
    random_seed: int

    @classmethod
    def load(cls) -> "PipelineSettings":
        """Load pipeline settings from environment variables or default locations."""
        settings = cls(
            data_raw_path=resolve_path(
                os.getenv("DATA_RAW_PATH"), get_default_raw_path()
            ),
            data_processed_path=resolve_path(
                os.getenv("DATA_PROCESSED_PATH"), get_default_processed_path()
            ),
            data_metadata_path=resolve_path(
                os.getenv("DATA_METADATA_PATH"), get_default_metadata_path()
            ),
            report_path=resolve_path(
                os.getenv("REPORT_PATH"), get_default_report_path()
            ),
            log_path=resolve_path(os.getenv("LOG_PATH"), get_default_log_path()),
            default_league=os.getenv("DEFAULT_LEAGUE", DEFAULT_LEAGUE),
            supported_leagues=SUPPORTED_LEAGUES,
            pipeline_version=os.getenv("PIPELINE_VERSION", PIPELINE_VERSION),
            random_seed=int(os.getenv("RANDOM_SEED", RANDOM_SEED)),
        )
        ensure_directories(
            [
                settings.data_raw_path,
                settings.data_processed_path,
                settings.data_metadata_path,
                settings.report_path,
                settings.log_path,
            ]
        )
        return settings
