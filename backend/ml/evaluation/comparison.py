"""
Model comparison for the evaluation module.

Ranks multiple trained models based on their evaluation metrics.
"""

from typing import List, Dict, Any, Tuple
from .exceptions import MissingMetricError

class ModelComparator:
    """Ranks multiple models based on evaluation metrics."""
    
    def __init__(self, primary_metric: str = "f1", higher_is_better: bool = True):
        """
        Initializes the ModelComparator.
        
        Args:
            primary_metric: The metric to use for ranking models. Default is "f1".
            higher_is_better: If True, higher metric values are ranked better.
        """
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better
        
    def rank_models(self, evaluation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ranks a list of evaluation results.
        
        Args:
            evaluation_results: A list of dicts. Each dict must contain a 'model_name' 
                                and a 'metrics' dictionary.
                                
        Returns:
            A list of the input dictionaries, sorted from best to worst.
            
        Raises:
            MissingMetricError: If the primary metric is missing in any result.
        """
        for result in evaluation_results:
            if "metrics" not in result or self.primary_metric not in result["metrics"]:
                model_name = result.get("model_name", "Unknown Model")
                raise MissingMetricError(
                    f"Model '{model_name}' is missing the primary metric '{self.primary_metric}'."
                )
                
        sorted_results = sorted(
            evaluation_results,
            key=lambda x: x["metrics"][self.primary_metric],
            reverse=self.higher_is_better
        )
        
        return sorted_results
