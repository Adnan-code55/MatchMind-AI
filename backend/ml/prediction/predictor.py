"""
Predictor facade for the Prediction module.
"""

import pandas as pd
from typing import Dict, Any, Tuple

from .pipeline import PredictionPipeline
from .metadata import PredictionMetadata


class MatchPredictor:
    """Main entry point for MatchMind AI predictions."""

    def __init__(self, model_directory: str = "models") -> None:
        """
        Initializes the MatchPredictor.
        
        Args:
            model_directory: The directory where models are stored.
        """
        self.pipeline = PredictionPipeline(model_directory=model_directory)

    def predict(self, match_data: pd.DataFrame) -> Tuple[Dict[str, float], float, PredictionMetadata]:
        """
        Predicts the outcome of a match.
        
        Args:
            match_data: A pandas DataFrame containing preprocessed features for a match.
            
        Returns:
            A tuple containing:
                - probabilities: Dictionary with keys 'home_win', 'draw', 'away_win'.
                - confidence_score: The confidence score of the prediction.
                - metadata: PredictionMetadata instance.
        """
        return self.pipeline.execute(match_data)
