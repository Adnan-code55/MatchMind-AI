"""
Metrics calculation for the evaluation module.

Wraps scikit-learn metrics to compute classification metrics.
"""

from typing import Dict, Optional, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
)

from .exceptions import MetricComputationError
from .validators import validate_arrays, validate_probabilities, validate_binary_labels


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes standard classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Prediction probabilities (required for ROC AUC and Log Loss).
        
    Returns:
        A dictionary containing computed metrics.
        
    Raises:
        MetricComputationError: If a metric fails to compute.
    """
    try:
        # Validate inputs
        validate_arrays(y_true, y_pred, y_prob)
        validate_binary_labels(y_true, y_pred)
        
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_prob is not None:
            validate_probabilities(y_prob)
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
            
        return metrics
    except Exception as e:
        raise MetricComputationError(f"Failed to compute classification metrics: {str(e)}") from e
