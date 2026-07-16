"""
Calculator utilities for the Feature Engineering Engine.

This module provides reusable functions to calculate specific football statistics
such as form, goal averages, win percentages, and goal differences.
"""

from typing import Union
import pandas as pd


def calculate_points(result: str, is_home: bool) -> int:
    """Calculate points earned in a match.
    
    Args:
        result: The match result ('H' for home win, 'A' for away win, 'D' for draw).
        is_home: True if the team was playing at home, False otherwise.
        
    Returns:
        Integer points (3 for win, 1 for draw, 0 for loss).
    """
    if result == 'D':
        return 1
    if (result == 'H' and is_home) or (result == 'A' and not is_home):
        return 3
    return 0


def calculate_is_win(result: str, is_home: bool) -> int:
    """Determine if the match was won by the team.
    
    Args:
        result: The match result ('H', 'A', 'D').
        is_home: True if the team was playing at home.
        
    Returns:
        1 if won, 0 otherwise.
    """
    if (result == 'H' and is_home) or (result == 'A' and not is_home):
        return 1
    return 0


def calculate_clean_sheet(goals_conceded: Union[int, float]) -> int:
    """Determine if the team kept a clean sheet.
    
    Args:
        goals_conceded: Number of goals conceded by the team.
        
    Returns:
        1 if 0 goals conceded, 0 otherwise.
    """
    return 1 if goals_conceded == 0 else 0


def compute_rolling_stats(
    team_history: pd.DataFrame, 
    window_size: int
) -> pd.DataFrame:
    """Compute rolling statistics for a team's history.
    
    This function assumes the history is sorted chronologically and calculates
    rolling averages/sums, ensuring the current match is excluded from its own
    prediction (using a shift).
    
    Args:
        team_history: DataFrame containing a team's past matches.
        window_size: The number of historical matches to include in the window.
        
    Returns:
        DataFrame containing the calculated rolling features.
    """
    # Shift by 1 to prevent target leakage (exclude current match from its own stats)
    shifted = team_history.shift(1)
    
    # Calculate rolling aggregations
    rolling = shifted.rolling(window=window_size, min_periods=1)
    
    # Generate features
    stats = pd.DataFrame(index=team_history.index)
    stats['Form'] = rolling['Points'].sum()
    stats['GoalsAverage'] = rolling['GoalsScored'].mean()
    stats['GoalsConcededAverage'] = rolling['GoalsConceded'].mean()
    stats['WinPercentage'] = rolling['IsWin'].mean()
    stats['CleanSheetPercentage'] = rolling['IsCleanSheet'].mean()
    stats['GoalDifference'] = rolling['GoalDifference'].mean()
    
    # Fill NaN values (for the first match of each team) with 0
    return stats.fillna(0)
