"""
MatchMind AI pipeline package.

This package exposes orchestration, reporting, versioning, metadata, and contract
validation capabilities for enterprise-grade data pipeline execution.
"""

from .orchestrator import PipelineOrchestrator
from .pipeline import Pipeline
from .report import PipelineReport
from .metadata import DatasetMetadata, LeagueMetadata
from .contracts import ContractValidator
from .integration import FeaturePipelineIntegration, IntegrationRunResult

__all__ = [
    "PipelineOrchestrator",
    "Pipeline",
    "PipelineReport",
    "DatasetMetadata",
    "LeagueMetadata",
    "ContractValidator",
    "FeaturePipelineIntegration",
    "IntegrationRunResult",
]
