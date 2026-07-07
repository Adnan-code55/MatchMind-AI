"""
MatchMind AI Data Pipeline - Milestone 1

Production-ready data foundation for football match machine learning.
"""

# Configuration and constants for the data pipeline
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TESTS_DIR = PROJECT_ROOT / "tests"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "pipeline.log"

# Data processing configuration
DEFAULT_TRAIN_SIZE = 0.8
DEFAULT_RANDOM_STATE = 42
DEFAULT_NORMALIZE = True
DEFAULT_ENCODE_CATEGORICAL = True

# Schema configuration
REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]

NUMERIC_COLUMNS = [
    "FTHG",
    "FTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]

CATEGORICAL_COLUMNS = ["HomeTeam", "AwayTeam", "FTR"]
