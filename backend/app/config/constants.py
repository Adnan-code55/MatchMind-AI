"""
Pipeline constants for MatchMind AI.

This module defines enterprise-grade defaults for league metadata, pipeline
versioning, league support, and operational behavior.
"""

from typing import Dict

DEFAULT_LEAGUE = "EPL"
PIPELINE_VERSION = "1.1.0"
RANDOM_SEED = 42

SUPPORTED_LEAGUES: Dict[str, Dict[str, str]] = {
    "EPL": {
        "name": "English Premier League",
        "country": "England",
        "competition": "Premier League",
        "data_provider": "Opta",
        "source": "Official Premier League",
    },
    "LA_LIGA": {
        "name": "LaLiga",
        "country": "Spain",
        "competition": "LaLiga",
        "data_provider": "Opta",
        "source": "Official LaLiga",
    },
}
