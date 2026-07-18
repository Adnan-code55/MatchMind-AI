"""
Prediction Pipeline for the Prediction module.
"""

import time
import pandas as pd
from typing import Dict, Any, Tuple

from backend.app.data.logger import PipelineLogger
from .exceptions import PredictionError
from .metadata import PredictionMetadata
from .loaders import ModelLoader
from .probability import ProbabilityGenerator
from .validators import validate_prediction_input


class PredictionPipeline:
    """Orchestrates the prediction process."""

    def __init__(self, model_directory: str = "models") -> None:
        """
        Initializes the PredictionPipeline.
        
        Args:
            model_directory: The directory where models are stored.
        """
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)
        self.model_loader = ModelLoader(directory=model_directory)
        self.probability_generator = ProbabilityGenerator()
        
        self.logger.info("Initializing PredictionPipeline.")
        self.model = self.model_loader.load_best_model()
        self.model_name = type(self.model).__name__

    def execute(self, df: pd.DataFrame) -> Tuple[Dict[str, float], float, PredictionMetadata]:
        """
        Executes the prediction pipeline.
        
        Args:
            df: The pandas DataFrame containing the input features for a single match.
            
        Returns:
            A tuple containing:
                - probabilities: Dictionary with keys 'home_win', 'draw', 'away_win'.
                - confidence_score: The confidence score of the prediction.
                - metadata: PredictionMetadata instance.
                
        Raises:
            PredictionError: If prediction fails.
        """
        self.logger.info("Executing prediction pipeline.")
        validate_prediction_input(df)

        start_time = time.time()
        
        try:
            # Check if model has predict_proba
            if not hasattr(self.model, "predict_proba"):
                raise PredictionError(f"Model {self.model_name} does not support probability predictions.")
                
            predict_proba_output = self.model.predict_proba(df)
            probs, confidence_score = self.probability_generator.generate_probabilities(predict_proba_output)
            
        except Exception as e:
            if isinstance(e, PredictionError):
                raise
            raise PredictionError(f"Failed to generate predictions: {e}")

        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000

        metadata = PredictionMetadata(
            model_name=self.model_name,
            inference_time_ms=inference_time_ms,
            confidence_score=confidence_score
        )
        
        self.logger.info(f"Prediction successful. Confidence: {confidence_score:.4f}")
        return probs, confidence_score, metadata
