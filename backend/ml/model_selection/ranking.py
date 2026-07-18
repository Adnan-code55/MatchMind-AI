"""
Ranking logic for the model selection module.
"""

from typing import List, Dict, Any
from backend.ml.evaluation.comparison import ModelComparator


class ModelRanker:
    """Ranks models based on evaluation metrics."""
    
    def __init__(self, primary_metric: str = "f1", higher_is_better: bool = True):
        """
        Initialize the ModelRanker.
        
        Args:
            primary_metric: Metric used to rank the models.
            higher_is_better: If True, higher metric values are ranked better.
        """
        self.comparator = ModelComparator(
            primary_metric=primary_metric,
            higher_is_better=higher_is_better
        )
        
    def rank(self, evaluation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank a list of evaluation results.
        
        Args:
            evaluation_results: List of dicts, each containing 'model_name' and 'metrics'.
            
        Returns:
            A list of ranked results from best to worst.
        """
        return self.comparator.rank_models(evaluation_results)
