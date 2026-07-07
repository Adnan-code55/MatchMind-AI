"""
Dataset metadata generation for MatchMind AI.

This module captures dataset versioning, checksum, and league metadata for each
processed dataset.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from ..config.constants import SUPPORTED_LEAGUES
from ..config.settings import PipelineSettings


@dataclass
class LeagueMetadata:
    """Describes league-level metadata for a dataset."""

    league: str
    league_name: str
    season: str
    country: str
    competition: str
    source: str
    data_provider: str

    def to_dict(self) -> Dict[str, str]:
        """Convert league metadata to a dictionary."""
        return {
            "league": self.league,
            "league_name": self.league_name,
            "season": self.season,
            "country": self.country,
            "competition": self.competition,
            "source": self.source,
            "data_provider": self.data_provider,
        }

    @classmethod
    def from_league_code(cls, league: str, season: str) -> "LeagueMetadata":
        """Create league metadata from a supported league code."""
        league_code = league.upper()
        metadata = SUPPORTED_LEAGUES.get(league_code)
        if metadata is None:
            raise ValueError(f"Unsupported league code: {league}")

        return cls(
            league=league_code,
            league_name=metadata["name"],
            season=season,
            country=metadata["country"],
            competition=metadata["competition"],
            source=metadata["source"],
            data_provider=metadata["data_provider"],
        )


@dataclass
class DatasetMetadata:
    """Versioned metadata for a processed dataset."""

    dataset_uuid: str
    version: int
    timestamp: str
    checksum: str
    pipeline_version: str
    rows: int
    columns: int
    league: str
    season: str
    country: str
    competition: str
    source_filename: str
    data_provider: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset metadata to a serializable dictionary."""
        return asdict(self)

    def save(self, directory: Path) -> Path:
        """Save the dataset metadata as JSON."""
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / f"{self.league}_{self.season}_v{self.version}.json"
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(self.to_dict(), metadata_file, indent=2)
        return metadata_path


def compute_file_checksum(file_path: Path) -> str:
    """Compute the SHA256 checksum for a file."""
    hash_algorithm = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            hash_algorithm.update(chunk)
    return hash_algorithm.hexdigest()


def next_dataset_version(directory: Path, league: str, season: str) -> int:
    """Calculate the next dataset version for a league and season."""
    directory.mkdir(parents=True, exist_ok=True)
    pattern = f"{league}_{season}_v*.json"
    existing_versions = []
    for metadata_file in directory.glob(pattern):
        stem = metadata_file.stem
        try:
            version_str = stem.split("_v")[-1]
            existing_versions.append(int(version_str))
        except ValueError:
            continue
    return max(existing_versions, default=0) + 1


def build_dataset_metadata(
    output_file: Path,
    source_filename: str,
    league_metadata: LeagueMetadata,
    settings: PipelineSettings,
    rows: int,
    columns: int,
) -> DatasetMetadata:
    """Build dataset metadata for a processed dataset."""
    version = next_dataset_version(
        settings.data_metadata_path,
        league_metadata.league,
        league_metadata.season,
    )
    return DatasetMetadata(
        dataset_uuid=str(uuid4()),
        version=version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        checksum=compute_file_checksum(output_file),
        pipeline_version=settings.pipeline_version,
        rows=rows,
        columns=columns,
        league=league_metadata.league,
        season=league_metadata.season,
        country=league_metadata.country,
        competition=league_metadata.competition,
        source_filename=source_filename,
        data_provider=league_metadata.data_provider,
    )
