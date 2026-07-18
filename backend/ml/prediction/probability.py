"""
Probability generation for the Prediction module.
"""

from typing import Dict, Tuple
import numpy as np


class ProbabilityGenerator:
    """Class responsible for mapping model predictions to probabilities and confidence."""

    def __init__(self, class_mapping: Dict[int, str] = None) -> None:
        """
        Initializes ProbabilityGenerator.
        
        Args:
            class_mapping: Optional custom mapping from class index to outcome name.
                           Default assumes indices 0, 1, 2 map to 'Home Win', 'Draw', 'Away Win'.
        """
        if class_mapping is None:
            self.class_mapping = {
                0: "home_win",
                1: "draw",
                2: "away_win"
            }
        else:
            self.class_mapping = class_mapping

    def generate_probabilities(self, predict_proba_output: np.ndarray) -> Tuple[Dict[str, float], float]:
        """
        Generates named probabilities and a confidence score from predict_proba output.
        
        Args:
            predict_proba_output: 1D array of probabilities for each class.
            
        Returns:
            A tuple of (probabilities_dict, confidence_score).
        """
        # Ensure it's a 1D array for a single prediction
        if predict_proba_output.ndim == 2:
            if predict_proba_output.shape[0] != 1:
                raise ValueError("Expected predict_proba_output for a single sample.")
            predict_proba_output = predict_proba_output[0]

        probs = {}
        for idx, prob in enumerate(predict_proba_output):
            if idx in self.class_mapping:
                probs[self.class_mapping[idx]] = float(prob)
        
        # Ensure we have all default keys even if model has fewer classes, though unlikely
        if "home_win" not in probs: probs["home_win"] = 0.0
        if "draw" not in probs: probs["draw"] = 0.0
        if "away_win" not in probs: probs["away_win"] = 0.0

        # Confidence score: max probability
        confidence_score = float(np.max(predict_proba_output))
        
        return probs, confidence_score
