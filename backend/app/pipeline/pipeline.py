"""
Pipeline orchestration for MatchMind AI.

This module defines the enterprise-grade pipeline that executes loading,
validation, cleaning, preprocessing, dataset persistence, report generation,
and metadata creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pandas as pd

from ..config.logging_config import configure_logging
from ..config.settings import PipelineSettings
from ..data.cleaner import DataCleaner
from ..data.data_loader import DataLoader
from ..data.exceptions import (
    DataValidationError,
    DatasetNotFoundError,
)
from ..data.logger import PipelineLogger
from ..data.preprocessor import DataPreprocessor
from ..data.validator import DataValidator
from .contracts import ContractValidator
from .metadata import build_dataset_metadata, LeagueMetadata
from .report import PipelineReport


@dataclass
class PipelineResult:
    """Result container for a pipeline execution."""

    report: PipelineReport
    output_file: Path
    metadata_file: Path
    log_file: Path


class Pipeline:
    """Enterprise pipeline for MatchMind AI dataset processing."""

    def __init__(
        self,
        settings: PipelineSettings,
        league: Optional[str] = None,
        season: Optional[str] = None,
        source_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> None:
        """Initialize the pipeline with runtime settings."""
        self.settings = settings
        self.league = (league or self.settings.default_league).upper()
        self.season = season or datetime.now(timezone.utc).strftime("%Y")
        self.source_path = Path(source_path) if source_path else self.settings.data_raw_path
        self.output_path = Path(output_path) if output_path else self.settings.data_processed_path
        self.execution_id = str(uuid4())
        self.start_time = datetime.now(timezone.utc)
        self._log_file = configure_logging(self.settings, self.execution_id)
        PipelineLogger.log_info(
            __name__,
            f"Pipeline initialized for league={self.league}, season={self.season}",
        )
        self.league_metadata = LeagueMetadata.from_league_code(self.league, self.season)
        self._contract_validator = ContractValidator()
        self._data_loader = DataLoader(self.source_path)
        self._validator = DataValidator()
        self._cleaner = DataCleaner()
        self._preprocessor = DataPreprocessor()

    def run(self) -> PipelineReport:
        """Run the pipeline end to end.

        Execution order:
            1. Load
            2. Validate
            3. Clean
            4. Preprocess
            5. Save
            6. Generate report
            7. Generate metadata
            8. Finish
        """
        validation_errors = []
        warnings: list[str] = []
        output_file = self.output_path / f"{self.league}_{self.season}_{self.execution_id}.csv"
        metadata_file: Optional[Path] = None
        report_file_path = self.settings.report_path / f"pipeline_report_{self.execution_id}.json"

        try:
            raw_df = self._load()
            warnings = self._validate(raw_df)
            cleaned_df = self._clean(raw_df)
            processed_df = self._preprocess(cleaned_df)
            self._save(processed_df, output_file)
            metadata = build_dataset_metadata(
                output_file=output_file,
                source_filename=self._source_filename(raw_df),
                league_metadata=self.league_metadata,
                settings=self.settings,
                rows=len(processed_df),
                columns=processed_df.shape[1],
            )
            metadata_file = metadata.save(self.settings.data_metadata_path)
            status = "success"
            PipelineLogger.log_info(__name__, "Pipeline execution completed successfully")
        except DataValidationError as error:
            validation_errors.append(str(error))
            status = "failed"
            PipelineLogger.log_error(__name__, f"Pipeline validation failed: {error}")
            raise
        except DatasetNotFoundError as error:
            validation_errors.append(str(error))
            status = "failed"
            PipelineLogger.log_error(__name__, f"Pipeline input source failed: {error}")
            raise
        except Exception as error:
            validation_errors.append(str(error))
            status = "failed"
            PipelineLogger.log_error(__name__, f"Pipeline execution failed: {error}")
            raise
        finally:
            finish_time = datetime.now(timezone.utc)
            duration_seconds = (finish_time - self.start_time).total_seconds()
            report = PipelineReport(
                execution_id=self.execution_id,
                start_time=self.start_time.isoformat(),
                finish_time=finish_time.isoformat(),
                duration_seconds=duration_seconds,
                rows_loaded=self._rows_loaded if hasattr(self, "_rows_loaded") else 0,
                rows_removed=self._rows_removed if hasattr(self, "_rows_removed") else 0,
                duplicate_rows=self._duplicate_rows if hasattr(self, "_duplicate_rows") else 0,
                missing_values_fixed=self._missing_values_fixed if hasattr(self, "_missing_values_fixed") else 0,
                validation_errors=validation_errors,
                warnings=warnings,
                output_file=str(output_file if output_file.exists() else ""),
                metadata_file=str(metadata_file) if metadata_file else "",
                report_file=str(report_file_path),
                pipeline_version=self.settings.pipeline_version,
                status=status,
                league=self.league,
                season=self.season,
                source=str(self.source_path),
            )
            report.save(self.settings.report_path)

        return report

    def _load(self) -> pd.DataFrame:
        """Load raw dataset(s) from the configured source path."""
        PipelineLogger.log_info(__name__, f"Loading dataset from {self.source_path}")
        df = self._data_loader.load_matches()
        self._rows_loaded = len(df)
        return df

    def _validate(self, df: pd.DataFrame) -> list[str]:
        """Validate raw dataset using contracts and schema validation."""
        PipelineLogger.log_info(__name__, "Validating dataset contracts")
        self._contract_validator.validate(df)
        PipelineLogger.log_info(__name__, "Contract validation passed")

        PipelineLogger.log_info(__name__, "Validating dataset schema and values")
        report = self._validator.validate(df)
        if report.warnings:
            PipelineLogger.log_warning(__name__, f"Validation warnings: {report.warnings}")
        return report.warnings

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the validated dataset."""
        PipelineLogger.log_info(__name__, "Cleaning dataset")
        initial_nulls = int(df.isna().sum().sum())
        initial_duplicates = int(df.duplicated().sum())
        cleaned_df = self._cleaner.clean(df)
        self._duplicate_rows = initial_duplicates
        self._missing_values_fixed = max(0, initial_nulls - int(cleaned_df.isna().sum().sum()))
        self._rows_removed = max(0, len(df) - len(cleaned_df))
        return cleaned_df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess cleaned dataset for downstream consumption."""
        PipelineLogger.log_info(__name__, "Preprocessing dataset")
        processed_df = self._preprocessor.encode_categorical_columns(df)
        processed_df = self._preprocessor.normalize_numeric_columns(processed_df)
        return processed_df

    def _save(self, df: pd.DataFrame, output_file: Path) -> None:
        """Save the processed dataset to disk."""
        PipelineLogger.log_info(__name__, f"Saving processed dataset to {output_file}")
        self._preprocessor.save_dataset(df, str(self.output_path), name=output_file.stem)

    def _source_filename(self, df: pd.DataFrame) -> str:
        """Extract source filename information for metadata."""
        if self.source_path.is_dir():
            return ", ".join(sorted([item.name for item in self.source_path.glob("*.csv")]))
        return self.source_path.name

