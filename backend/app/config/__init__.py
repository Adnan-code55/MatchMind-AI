"""
MatchMind AI configuration package.

This package exposes the centralized settings and path utilities used by the
pipeline, report generation, logging, and metadata services.
"""

from .settings import PipelineSettings
from .constants import DEFAULT_LEAGUE, SUPPORTED_LEAGUES, PIPELINE_VERSION
from .paths import (
    get_project_root,
    get_default_raw_path,
    get_default_processed_path,
    get_default_metadata_path,
    get_default_report_path,
    get_default_log_path,
)

__all__ = [
    "PipelineSettings",
    "DEFAULT_LEAGUE",
    "SUPPORTED_LEAGUES",
    "PIPELINE_VERSION",
    "get_project_root",
    "get_default_raw_path",
    "get_default_processed_path",
    "get_default_metadata_path",
    "get_default_report_path",
    "get_default_log_path",
]
