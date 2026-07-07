"""
MatchMind AI feature engineering package.

This package exposes the architecture for feature generation, registration,
pipeline execution, and feature combination.
"""

from .base import FeatureGenerator
from .combiner import FeatureCombiner
from .pipeline import FeaturePipeline
from .registry import FeatureRegistry
import importlib
from typing import Any

# Lazy import mapping: attribute name -> module path
_LAZY_MODULES: dict[str, str] = {
    "GoalDifferenceGenerator": "backend.app.features.goal_difference",
    "RecentFormGenerator": "backend.app.features.recent_form",
    "TeamPerformanceGenerator": "backend.app.features.team_performance",
    "HomeAdvantageGenerator": "backend.app.features.home_advantage",
    "RestDaysGenerator": "backend.app.features.rest_days",
    "HeadToHeadGenerator": "backend.app.features.head_to_head_generator",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - simple lazy loader
    """Lazy-load feature generator classes on attribute access.

    This keeps package import fast and avoids importing submodules until
    explicitly requested, while still exposing generator classes via the
    package namespace.
    """
    module_path = _LAZY_MODULES.get(name)
    if not module_path:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_MODULES.keys()))
