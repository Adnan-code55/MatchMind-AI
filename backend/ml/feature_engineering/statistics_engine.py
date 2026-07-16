"""
Statistics Engine for Feature Engineering.

This module provides the StatisticsEngine class which orchestrates the generation
of historical rolling statistics from match data.
"""

from typing import Tuple
import pandas as pd

from backend.app.data.logger import PipelineLogger
from .exceptions import InvalidWindowError
from .metadata import FeatureMetadata
from .validators import validate_required_columns
from .calculators import (
    calculate_points,
    calculate_is_win,
    calculate_clean_sheet,
    compute_rolling_stats,
)


class StatisticsEngine:
    """Engine for generating rolling football statistics.

    Processes historical match data to compute features such as team form,
    goal averages, and win percentages over a configurable rolling window.
    """

    REQUIRED_COLUMNS = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    ]

    def __init__(self, window_size: int = 5) -> None:
        """Initialize the StatisticsEngine.

        Args:
            window_size: The number of historical matches to include in the rolling window.
                Defaults to 5.

        Raises:
            InvalidWindowError: If window_size is less than 1.
        """
        if window_size < 1:
            raise InvalidWindowError(f"Window size must be at least 1, got {window_size}")
        
        self.window_size = window_size
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)

    def generate_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, FeatureMetadata]:
        """Generate rolling statistical features for the dataset.

        Args:
            df: The input pandas DataFrame containing match history.

        Returns:
            A tuple containing:
            - The DataFrame with appended historical features.
            - FeatureMetadata describing the operation.
            
        Raises:
            MissingColumnError: If required columns are missing from the input dataset.
        """
        self.logger.info(f"Generating features with window size {self.window_size}")
        
        validate_required_columns(df, self.REQUIRED_COLUMNS)
        
        initial_features = len(df.columns)
        
        # We need a stable chronological order for rolling statistics
        if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df = df.sort_values("Date").reset_index(drop=True)
            
        # Transform the dataset into a team-level view
        team_stats = self._build_team_history(df)
        
        # Compute rolling stats per team
        rolling_stats = (
            team_stats.groupby("Team", group_keys=False)
            .apply(lambda group: compute_rolling_stats(group, self.window_size))
        )
        
        # Merge back to original DataFrame
        result_df = self._merge_stats(df, rolling_stats, team_stats)
        
        new_features = [
            "Home Team Form", "Away Team Form",
            "Home Goals Average", "Away Goals Average",
            "Home Goals Conceded Average", "Away Goals Conceded Average",
            "Home Win Percentage", "Away Win Percentage",
            "Home Clean Sheet Percentage", "Away Clean Sheet Percentage",
            "Home Goal Difference", "Away Goal Difference"
        ]
        
        final_features = len(result_df.columns)
        
        metadata = FeatureMetadata(
            window_size=self.window_size,
            features_generated=new_features,
            initial_feature_count=initial_features,
            final_feature_count=final_features,
        )
        
        self.logger.info(f"Feature engineering complete. Added {len(new_features)} features.")
        return result_df, metadata

    def _build_team_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restructure match data into a chronologically ordered team history.
        
        Args:
            df: Original match dataset.
            
        Returns:
            A DataFrame where each row represents a team's performance in a match.
        """
        # Home view
        home_df = pd.DataFrame({
            "MatchIndex": df.index,
            "Date": df["Date"],
            "Team": df["HomeTeam"],
            "IsHome": True,
            "GoalsScored": df["FTHG"],
            "GoalsConceded": df["FTAG"],
            "Result": df["FTR"],
        })
        
        # Away view
        away_df = pd.DataFrame({
            "MatchIndex": df.index,
            "Date": df["Date"],
            "Team": df["AwayTeam"],
            "IsHome": False,
            "GoalsScored": df["FTAG"],
            "GoalsConceded": df["FTHG"],
            "Result": df["FTR"],
        })
        
        team_history = pd.concat([home_df, away_df]).sort_values(by=["Date", "MatchIndex"]).reset_index(drop=True)
        
        # Calculate raw match-level stats
        team_history["Points"] = team_history.apply(
            lambda x: calculate_points(x["Result"], x["IsHome"]), axis=1
        )
        team_history["IsWin"] = team_history.apply(
            lambda x: calculate_is_win(x["Result"], x["IsHome"]), axis=1
        )
        team_history["IsCleanSheet"] = team_history["GoalsConceded"].apply(calculate_clean_sheet)
        team_history["GoalDifference"] = team_history["GoalsScored"] - team_history["GoalsConceded"]
        
        return team_history

    def _merge_stats(
        self, 
        df: pd.DataFrame, 
        rolling_stats: pd.DataFrame, 
        team_history: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge the calculated rolling statistics back onto the main match dataset.
        
        Args:
            df: Original match dataset.
            rolling_stats: Calculated rolling statistics per team per match.
            team_history: The restructured team history dataset.
            
        Returns:
            The augmented match dataset.
        """
        # Attach the MatchIndex back to rolling stats to join with original df
        rolling_stats["MatchIndex"] = team_history["MatchIndex"]
        rolling_stats["IsHome"] = team_history["IsHome"]
        
        # Separate home and away stats
        home_stats = rolling_stats[rolling_stats["IsHome"] == True].copy()
        away_stats = rolling_stats[rolling_stats["IsHome"] == False].copy()
        
        home_stats = home_stats.set_index("MatchIndex").drop(columns=["IsHome"])
        away_stats = away_stats.set_index("MatchIndex").drop(columns=["IsHome"])
        
        # Rename columns
        home_mapping = {
            "Form": "Home Team Form",
            "GoalsAverage": "Home Goals Average",
            "GoalsConcededAverage": "Home Goals Conceded Average",
            "WinPercentage": "Home Win Percentage",
            "CleanSheetPercentage": "Home Clean Sheet Percentage",
            "GoalDifference": "Home Goal Difference",
        }
        
        away_mapping = {
            "Form": "Away Team Form",
            "GoalsAverage": "Away Goals Average",
            "GoalsConcededAverage": "Away Goals Conceded Average",
            "WinPercentage": "Away Win Percentage",
            "CleanSheetPercentage": "Away Clean Sheet Percentage",
            "GoalDifference": "Away Goal Difference",
        }
        
        home_stats = home_stats.rename(columns=home_mapping)
        away_stats = away_stats.rename(columns=away_mapping)
        
        # Merge back
        result = df.copy()
        result = result.join(home_stats)
        result = result.join(away_stats)
        
        return result
