"""
Core model selector for the ML module.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd

from backend.app.data.logger import PipelineLogger
from backend.ml.training.models import get_supported_models
from backend.ml.training.trainer import ModelTrainer
from backend.ml.evaluation.evaluator import EvaluationEngine
from backend.ml.evaluation.metadata import ModelMetadata

from .ranking import ModelRanker
from .persistence import persist_best_model
from .metadata import ModelSelectionMetadata
from .validators import validate_models_list, validate_selection_dataset
from .exceptions import NoModelsTrainedError

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Coordinates the automated training, evaluation, ranking, and selection
    of the best machine learning model.
    """
    
    def __init__(self, primary_metric: str = "f1", higher_is_better: bool = True, random_seed: int = 42):
        """
        Initialize the ModelSelector.
        
        Args:
            primary_metric: Metric used to select the best model.
            higher_is_better: Whether a higher metric value is better.
            random_seed: Seed for reproducibility.
        """
        self.primary_metric = primary_metric
        self.random_seed = random_seed
        self.ranker = ModelRanker(primary_metric=primary_metric, higher_is_better=higher_is_better)
        self.evaluator = EvaluationEngine()
        self.logger = PipelineLogger.get_logger(self.__class__.__name__)
        
    def run_selection(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        dataset_name: str = "Unknown Dataset"
    ) -> Tuple[Any, ModelSelectionMetadata]:
        """
        Trains all supported models, evaluates them, ranks them, selects the best,
        and persists it to disk.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_validation: Validation features.
            y_validation: Validation target.
            dataset_name: Name of the dataset for metadata purposes.
            
        Returns:
            A tuple containing (best_model_instance, selection_metadata).
            
        Raises:
            NoModelsTrainedError: If no models could be successfully trained and evaluated.
        """
        self.logger.info("Starting model selection process.")
        
        models_to_train = get_supported_models()
        validate_models_list(models_to_train)
        validate_selection_dataset(X_train, y_train, X_validation, y_validation)
        
        evaluation_results = []
        trained_models = {}
        
        for model_name in models_to_train:
            self.logger.info(f"Training and evaluating model: {model_name}")
            try:
                # Train
                trainer = ModelTrainer(random_seed=self.random_seed)
                trainer.train(model_name, X_train, y_train, X_validation, y_validation)
                trained_model = trainer.model
                
                trained_models[model_name] = trained_model
                
                # Predict on validation set
                y_pred = trained_model.predict(X_validation)
                y_prob = None
                if hasattr(trained_model, "predict_proba"):
                    probs = trained_model.predict_proba(X_validation)
                    if probs.shape[1] == 2:
                        y_prob = probs[:, 1]
                    else:
                        y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                        
                # Evaluate using existing evaluation framework
                model_meta = ModelMetadata(name=model_name, version="latest")
                metrics, eval_meta = self.evaluator.evaluate_predictions(
                    y_true=y_validation.values if isinstance(y_validation, pd.Series) else y_validation,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    model_metadata=model_meta,
                    dataset_name=dataset_name,
                    split_strategy="ValidationSplit"
                )
                
                evaluation_results.append({
                    "model_name": model_name,
                    "metrics": metrics
                })
                self.logger.info(f"Model {model_name} evaluated successfully.")
                
            except Exception as e:
                self.logger.error(f"Failed to train or evaluate model {model_name}: {str(e)}")
                
        if not evaluation_results:
            raise NoModelsTrainedError("No models were successfully trained and evaluated.")
            
        # Rank models
        self.logger.info("Ranking models...")
        ranked_results = self.ranker.rank(evaluation_results)
        
        best_result = ranked_results[0]
        best_model_name = best_result["model_name"]
        best_model = trained_models[best_model_name]
        
        self.logger.info(f"Best model selected: {best_model_name} with {self.primary_metric}={best_result['metrics'][self.primary_metric]:.4f}")
        
        # Generate metadata
        selection_metadata = ModelSelectionMetadata(
            winning_model=best_model_name,
            evaluation_metrics=best_result["metrics"],
            ranking=ranked_results,
            extra={"dataset_name": dataset_name, "primary_metric": self.primary_metric}
        )
        
        # Persist best model
        model_path, metadata_path = persist_best_model(best_model, selection_metadata)
        self.logger.info(f"Best model persisted to {model_path} and metadata to {metadata_path}")
        
        return best_model, selection_metadata
