"""
Core evaluation engine for MatchMind AI.

Provides the EvaluationEngine class to orchestrate validation, metrics computation,
and metadata generation. Supports KFold and StratifiedKFold evaluation.
"""

import logging
from typing import Optional, Dict, Any, Tuple, Generator, List
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, clone

from .metrics import compute_classification_metrics
from .metadata import EvaluationMetadata, ModelMetadata

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Orchestrates model evaluation, metrics computation, and metadata generation."""
    
    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        model_metadata: ModelMetadata,
        dataset_name: str,
        split_strategy: str = "SingleSplit",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], EvaluationMetadata]:
        """
        Evaluates predictions against true labels.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Prediction probabilities (optional).
            model_metadata: Metadata about the model.
            dataset_name: Identifier for the dataset.
            split_strategy: Strategy used to generate predictions.
            extra_metadata: Additional context for the evaluation run.
            
        Returns:
            A tuple of (metrics_dict, EvaluationMetadata).
        """
        logger.info(f"Evaluating predictions for model {model_metadata.name} on dataset {dataset_name}.")
        
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        
        metadata = EvaluationMetadata(
            model=model_metadata,
            dataset_name=dataset_name,
            num_samples=len(y_true),
            split_strategy=split_strategy,
            extra=extra_metadata or {}
        )
        
        return metrics, metadata
        
    def cross_validate(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        stratified: bool = True,
        random_state: Optional[int] = None,
        model_metadata: Optional[ModelMetadata] = None,
        dataset_name: str = "Unknown Dataset"
    ) -> Tuple[Dict[str, Any], EvaluationMetadata]:
        """
        Evaluates a scikit-learn compatible model using cross-validation.
        Aggregates metrics across folds by averaging them.
        
        Args:
            model: An unfitted scikit-learn estimator.
            X: Feature matrix.
            y: Target array.
            n_splits: Number of CV folds.
            stratified: Whether to use StratifiedKFold or standard KFold.
            random_state: Seed for the CV splitter.
            model_metadata: Metadata about the model. If None, generated automatically.
            dataset_name: Identifier for the dataset.
            
        Returns:
            A tuple of (aggregated_metrics_dict, EvaluationMetadata).
        """
        if stratified:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_strategy = "StratifiedKFold"
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_strategy = "KFold"
            
        if model_metadata is None:
            model_name = getattr(model, "__class__", type(model)).__name__
            model_metadata = ModelMetadata(name=model_name, version="unknown")
            
        logger.info(f"Starting {n_splits}-fold {split_strategy} for model {model_metadata.name}.")
        
        metrics_list: List[Dict[str, Any]] = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            y_pred = fold_model.predict(X_test)
            y_prob = None
            if hasattr(fold_model, "predict_proba"):
                # Use probabilities for the positive class (assuming binary classification)
                probs = fold_model.predict_proba(X_test)
                if probs.shape[1] == 2:
                    y_prob = probs[:, 1]
                else:
                    # Multiclass not currently fully supported in standard roc_auc without args,
                    # but fallback to binary assuming class 1 is index 1.
                    y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                    
            fold_metrics = compute_classification_metrics(y_test, y_pred, y_prob)
            metrics_list.append(fold_metrics)
            
        # Aggregate metrics
        aggregated_metrics = {}
        # Get all keys except confusion_matrix
        keys = [k for k in metrics_list[0].keys() if k != "confusion_matrix"]
        
        for key in keys:
            aggregated_metrics[key] = float(np.mean([m[key] for m in metrics_list]))
            
        # For confusion matrix, sum across folds
        if "confusion_matrix" in metrics_list[0]:
            cm_sum = np.sum([m["confusion_matrix"] for m in metrics_list], axis=0)
            aggregated_metrics["confusion_matrix"] = cm_sum.tolist()
            
        metadata = EvaluationMetadata(
            model=model_metadata,
            dataset_name=dataset_name,
            num_samples=len(y),
            split_strategy=f"{n_splits}-fold {split_strategy}",
            extra={"n_splits": n_splits, "stratified": stratified}
        )
        
        return aggregated_metrics, metadata
