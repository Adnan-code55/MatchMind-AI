"""
Integration milestone 3.1.5 for MatchMind AI.

This module orchestrates end-to-end feature engineering validation by running
all registered feature generators, collecting per-generator metadata,
generating manifest and quality reports, and validating temporal integrity.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..data.logger import PipelineLogger
from ..features.combiner import FeatureCombiner
from ..features.pipeline import FeaturePipeline
from ..features.registry import FeatureRegistry
from ..validation.feature_validator import FeatureValidator


class IntegrationValidationError(Exception):
    """Raised when an integration validation step fails."""


@dataclass
class FeatureManifestEntry:
    """Metadata about a single feature generator execution."""

    generator: str
    required_columns: List[str]
    output_columns: List[str]
    status: str
    rows_processed: int
    generated_feature_count: int
    execution_seconds: float
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureManifest:
    """Collection of feature generator execution metadata."""

    generators: List[FeatureManifestEntry]

    def to_dict(self) -> Dict[str, Any]:
        return {"generators": [entry.to_dict() for entry in self.generators]}

    def save(self, directory: Path, filename: str = "feature_manifest.json") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path


@dataclass
class DatasetQualityReport:
    """Summary of dataset quality after feature generation."""

    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    generated_feature_count: int
    fully_null_features: List[str]
    duplicate_rows: int
    missing_value_summary: Dict[str, int]
    critical_issues: List[str]
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, directory: Path, filename: str = "dataset_quality_report.json") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path


@dataclass
class PipelineBenchmarkReport:
    """Execution timings for the feature engineering pipeline."""

    total_duration_seconds: float
    generator_benchmarks: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, directory: Path, filename: str = "pipeline_benchmark.json") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path


@dataclass
class TimeIntegrityReport:
    """Report describing temporal integrity checks for feature generators."""

    passed: bool
    checked_generators: List[str]
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, directory: Path, filename: str = "time_integrity_report.json") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path


@dataclass
class IntegrationRunResult:
    """Container for all integration outputs."""

    manifest: FeatureManifest
    dataset_quality_report: DatasetQualityReport
    benchmark_report: PipelineBenchmarkReport
    time_integrity_report: TimeIntegrityReport
    validation_report: Dict[str, Any]
    output_dataframe: pd.DataFrame

    def save_reports(self, directory: Path) -> Dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        return {
            "feature_manifest": self.manifest.save(directory),
            "dataset_quality_report": self.dataset_quality_report.save(directory),
            "pipeline_benchmark": self.benchmark_report.save(directory),
            "time_integrity_report": self.time_integrity_report.save(directory),
        }


class FeaturePipelineIntegration:
    """Run the feature engineering pipeline and generate integration reports."""

    def __init__(
        self,
        registry: Optional[FeatureRegistry] = None,
        combiner: Optional[FeatureCombiner] = None,
        validator: Optional[FeatureValidator] = None,
        feature_order: Optional[List[str]] = None,
    ) -> None:
        self.registry = registry or FeatureRegistry()
        self.combiner = combiner or FeatureCombiner()
        self.validator = validator or FeatureValidator()
        self.feature_order = feature_order
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)

    def run(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        output_directory: Optional[Path] = None,
    ) -> IntegrationRunResult:
        """Execute the feature pipeline end to end and build integration reports."""
        if not isinstance(df, pd.DataFrame):
            raise IntegrationValidationError("Input must be a pandas DataFrame.")

        working_df = df.copy()
        if "Date" in working_df.columns:
            working_df["Date"] = pd.to_datetime(working_df["Date"])
            working_df = working_df.sort_values("Date").reset_index(drop=True)

        self.registry.reset()
        pipeline = FeaturePipeline(
            registry=self.registry,
            combiner=self.combiner,
            feature_order=self.feature_order,
        )

        manifest_entries: List[FeatureManifestEntry] = []
        current_df = working_df.copy()
        start_time = time.perf_counter()

        for generator in pipeline.generators:
            generator_start = time.perf_counter()
            try:
                generator.validate_input(current_df)
                feature_df = generator.generate(current_df)
            except Exception as exc:  # pragma: no cover - defensive path
                raise IntegrationValidationError(
                    f"Feature generator '{generator.name}' failed: {exc}"
                ) from exc

            elapsed = time.perf_counter() - generator_start
            if not isinstance(feature_df, pd.DataFrame):
                raise IntegrationValidationError(
                    f"Feature generator '{generator.name}' returned an invalid type."
                )
            if len(feature_df) != len(current_df):
                raise IntegrationValidationError(
                    f"Feature generator '{generator.name}' changed row count from {len(current_df)} to {len(feature_df)}."
                )

            feature_df = self._prefix_feature_columns(feature_df, generator.name)
            current_df = self.combiner.combine(current_df, feature_df)
            manifest_entries.append(
                FeatureManifestEntry(
                    generator=generator.name,
                    required_columns=list(generator.required_columns),
                    output_columns=list(feature_df.columns),
                    status="passed",
                    rows_processed=len(current_df),
                    generated_feature_count=len(feature_df.columns),
                    execution_seconds=round(elapsed, 6),
                )
            )

        total_duration = round(time.perf_counter() - start_time, 6)
        input_columns = list(working_df.columns)
        generated_feature_columns = [
            column for column in current_df.columns if column not in input_columns
        ]

        validation_df = current_df[generated_feature_columns].copy()
        if validation_df.empty:
            validation_df = current_df.copy()

        validation_report = self.validator.validate(validation_df, target_column=target_column)
        validation_dict = validation_report.to_dict()
        non_critical_warning_keywords = (
            "constant features",
            "near-constant",
            "highly imbalanced",
            "target distribution",
            "duplicate row",
            "duplicate columns",
        )

        critical_validation_warnings = [
            warning
            for warning in validation_report.warnings
            if not any(keyword in warning.lower() for keyword in non_critical_warning_keywords)
        ]
        validation_dict["is_valid"] = not critical_validation_warnings

        missing_value_summary = current_df[generated_feature_columns].isna().sum().to_dict()
        fully_null_features = [
            column for column in generated_feature_columns if current_df[column].isna().all()
        ]
        critical_issues = []
        if fully_null_features:
            critical_issues.append(
                f"Generated features entirely null: {fully_null_features}"
            )
        critical_issues.extend(critical_validation_warnings)

        quality_report = DatasetQualityReport(
            input_rows=len(working_df),
            output_rows=len(current_df),
            input_columns=len(working_df.columns),
            output_columns=len(current_df.columns),
            generated_feature_count=len(generated_feature_columns),
            fully_null_features=fully_null_features,
            duplicate_rows=int(current_df.duplicated().sum()),
            missing_value_summary={key: int(value) for key, value in missing_value_summary.items()},
            critical_issues=critical_issues,
            is_valid=not critical_issues,
        )

        benchmark_report = PipelineBenchmarkReport(
            total_duration_seconds=total_duration,
            generator_benchmarks=[entry.to_dict() for entry in manifest_entries],
        )

        time_integrity_report = self._validate_time_integrity(working_df, pipeline.generators)

        result = IntegrationRunResult(
            manifest=FeatureManifest(generators=manifest_entries),
            dataset_quality_report=quality_report,
            benchmark_report=benchmark_report,
            time_integrity_report=time_integrity_report,
            validation_report=validation_dict,
            output_dataframe=current_df,
        )

        if output_directory is not None:
            result.save_reports(output_directory)

        self.logger.info("Integration milestone 3.1.5 completed successfully")
        return result

    def _validate_time_integrity(
        self,
        df: pd.DataFrame,
        generators: List[Any],
    ) -> TimeIntegrityReport:
        """Validate that generators behave chronologically and do not use future data."""
        if df.empty:
            return TimeIntegrityReport(
                passed=False,
                checked_generators=[],
                issues=["Input dataset is empty."],
            )

        issues: List[str] = []
        checked_generators: List[str] = []

        for generator in generators:
            checked_generators.append(generator.name)
            try:
                head_df = df.head(1).copy()
                if "Date" in head_df.columns:
                    head_df["Date"] = pd.to_datetime(head_df["Date"])
                features_df = generator.generate(head_df)
            except Exception as exc:
                issues.append(f"{generator.name} failed time-integrity check: {exc}")
                continue

            if not isinstance(features_df, pd.DataFrame):
                issues.append(f"{generator.name} returned a non-DataFrame during time-integrity check")
                continue

            if features_df.empty:
                continue

            numeric_values = features_df.fillna(0.0)
            if not numeric_values.eq(0.0).all().all():
                issues.append(
                    f"{generator.name} produced non-zero values for the first row, suggesting future-data usage."
                )

        return TimeIntegrityReport(
            passed=not issues,
            checked_generators=checked_generators,
            issues=issues,
        )

    def _prefix_feature_columns(self, feature_df: pd.DataFrame, generator_name: str) -> pd.DataFrame:
        """Prefix generated feature columns to avoid collisions across generators."""
        prefixed_df = feature_df.copy()
        prefixed_df.columns = [f"{generator_name}::{column}" for column in prefixed_df.columns]
        return prefixed_df
