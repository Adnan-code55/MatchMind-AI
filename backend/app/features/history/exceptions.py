"""
Custom exceptions for the History Engine.

This module defines exceptions specific to history operations,
including errors related to invalid queries, missing data, and cache operations.
"""

from ...data.exceptions import MatchMindAIException


class HistoryEngineException(MatchMindAIException):
    """Base exception for all History Engine operations."""

    pass


class InvalidTeamError(HistoryEngineException):
    """Raised when a team name is invalid or not found."""

    pass


class InvalidDateError(HistoryEngineException):
    """Raised when date filtering produces invalid results."""

    pass


class NoHistoryFoundError(HistoryEngineException):
    """Raised when requested historical data does not exist."""

    pass


class CacheError(HistoryEngineException):
    """Raised when cache operations fail."""

    pass
