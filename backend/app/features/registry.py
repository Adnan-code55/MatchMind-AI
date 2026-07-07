"""
Registry and discovery support for MatchMind AI feature generators.

This module manages feature generator registration, instantiation, and optional
module discovery for the feature engineering engine.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Type

from .base import FeatureGenerator


class FeatureRegistry:
    """Registry for available feature generator implementations."""

    _registry: ClassVar[Dict[str, Type[FeatureGenerator]]] = {}

    @classmethod
    def register(cls, generator_cls: Type[FeatureGenerator]) -> Type[FeatureGenerator]:
        """Register a feature generator implementation."""
        if not issubclass(generator_cls, FeatureGenerator):
            raise TypeError("Only FeatureGenerator subclasses can be registered.")

        name = getattr(generator_cls, "name", generator_cls.__name__)
        if name in cls._registry:
            raise ValueError(f"Feature generator '{name}' is already registered.")

        cls._registry[name] = generator_cls
        return generator_cls

    @classmethod
    def get(cls, name: str) -> Type[FeatureGenerator]:
        """Retrieve a registered generator by its name."""
        try:
            return cls._registry[name]
        except KeyError as error:
            raise KeyError(f"Feature generator '{name}' is not registered.") from error

    @classmethod
    def list_generators(cls) -> List[str]:
        """Return a list of registered generator names."""
        return sorted(cls._registry.keys())

    @classmethod
    def instantiate_all(
        cls,
        configs: Optional[Dict[str, Dict[str, Any]]] = None,
        order: Optional[List[str]] = None,
    ) -> List[FeatureGenerator]:
        """Instantiate registered generators, optionally in a specific order."""
        configs = configs or {}
        generator_names = order or cls.list_generators()

        instances: List[FeatureGenerator] = []
        for name in generator_names:
            generator_cls = cls.get(name)
            generator_config = configs.get(name, {})
            instances.append(generator_cls(config=generator_config))

        return instances

    @classmethod
    def discover_generators(cls, package: Optional[str] = None) -> List[str]:
        """Discover and import generator modules from the features package."""
        package_name = package or __name__.rsplit(".", 1)[0]
        package_path = Path(importlib.import_module(package_name).__file__).parent

        for module_path in sorted(package_path.glob("*.py")):
            if module_path.name.startswith("_"):
                continue
            module_name = module_path.stem
            if module_name == "__init__":
                continue
            importlib.import_module(f"{package_name}.{module_name}")

        return cls.list_generators()

    @classmethod
    def reset(cls) -> None:
        """Reset the registry, removing all registered generators."""
        cls._registry.clear()
