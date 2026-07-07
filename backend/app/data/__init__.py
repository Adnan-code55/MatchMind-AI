"""
MatchMind AI Data Pipeline Package.

This package provides a complete, production-ready data pipeline for processing
football match data. It includes modules for loading, validating, cleaning,
and preprocessing data for machine learning.
"""

from .schema import FootballMatchSchema, DataType
from .exceptions import (
    MatchMindAIException,
    DataValidationError,
    MissingColumnError,
    DuplicateRowError,
    NullValueError,
    InvalidDateError,
    InvalidScoreError,
    InvalidTeamNameError,
    DatasetNotFoundError,
    InvalidDatasetError,
    SchemaMismatchError,
    DataProcessingError,
)
from .logger import PipelineLogger, PipelineFormatter
from .data_loader import DataLoader
from .validator import DataValidator, ValidationReport
from .cleaner import DataCleaner
from .preprocessor import DataPreprocessor

__version__ = "1.0.0"
__author__ = "MatchMind AI"
__all__ = [
    "FootballMatchSchema",
    "DataType",
    "MatchMindAIException",
    "DataValidationError",
    "MissingColumnError",
    "DuplicateRowError",
    "NullValueError",
    "InvalidDateError",
    "InvalidScoreError",
    "InvalidTeamNameError",
    "DatasetNotFoundError",
    "InvalidDatasetError",
    "SchemaMismatchError",
    "DataProcessingError",
    "PipelineLogger",
    "PipelineFormatter",
    "DataLoader",
    "DataValidator",
    "ValidationReport",
    "DataCleaner",
    "DataPreprocessor",
]
