"""
Reusable historical data access for MatchMind AI feature generation.

The History Engine is the single source of chronological football match history
for current and future feature generators.
"""

from .cache import HistoryCache
from .head_to_head import HeadToHeadHistory
from .match_history import MatchHistory
from .team_history import TeamHistory

__all__ = [
    "HistoryCache",
    "HeadToHeadHistory",
    "MatchHistory",
    "TeamHistory",
]
