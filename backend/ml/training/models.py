"""
Model factory for the ML Training module.
"""
from typing import Any, Dict
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .exceptions import UnsupportedModelError

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}

def get_supported_models() -> list[str]:
    """Return a list of supported model names."""
    return list(MODEL_REGISTRY.keys())

def build_model(model_name: str, random_seed: int, **kwargs: Any) -> ClassifierMixin:
    """
    Instantiate an sklearn model by name.
    
    Args:
        model_name: Name of the model.
        random_seed: Random seed for reproducibility.
        **kwargs: Additional hyperparameters for the model.
        
    Returns:
        An unfitted sklearn estimator.
        
    Raises:
        UnsupportedModelError: If the model name is not in the registry.
    """
    model_cls = MODEL_REGISTRY.get(model_name)
    if not model_cls:
        raise UnsupportedModelError(
            f"Model type '{model_name}' is not supported. Supported models: {get_supported_models()}"
        )

    params: Dict[str, Any] = dict(kwargs)
    
    # Inject random_state if the model supports it
    import inspect
    sig = inspect.signature(model_cls.__init__)
    if "random_state" in sig.parameters:
        params["random_state"] = random_seed
        
    return model_cls(**params)
