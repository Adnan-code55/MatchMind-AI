"""
Path utilities for MatchMind AI.

This module manages default project paths for raw data, processed outputs,
metadata, reports, and logs.
"""

from pathlib import Path
from typing import Optional, Union


def get_project_root() -> Path:
    """Get the repository root for the MatchMind AI project."""
    return Path(__file__).resolve().parents[3]


def resolve_path(
    path: Optional[Union[str, Path]], default_path: Path
) -> Path:
    """Resolve configured path or fall back to the default path."""
    if path is None:
        return default_path

    resolved_path = Path(path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = get_project_root() / resolved_path

    return resolved_path.resolve()


def get_default_raw_path() -> Path:
    """Return the default raw data directory."""
    return get_project_root() / "data" / "raw"


def get_default_processed_path() -> Path:
    """Return the default processed data directory."""
    return get_project_root() / "data" / "processed"


def get_default_metadata_path() -> Path:
    """Return the default metadata directory."""
    return get_project_root() / "metadata"


def get_default_report_path() -> Path:
    """Return the default report directory."""
    return get_project_root() / "reports"


def get_default_log_path() -> Path:
    """Return the default log directory."""
    return get_project_root() / "logs"


def ensure_directories(paths: list[Path]) -> None:
    """Ensure directories exist for the configured paths."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
