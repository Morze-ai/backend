"""Provides decorator-based experiment registration and factory-based experiment instantiation by name."""

from __future__ import annotations

from typing import TypeVar

from src.experiments.base import BaseExperiment

ExperimentType = TypeVar("ExperimentType", bound=type[BaseExperiment])

_REGISTRY: dict[str, type[BaseExperiment]] = {}
_ALIASES: dict[str, str] = {
    "mlp": "mlp_classifier",
    "linear": "linear_classifier",
    "logistic": "logistic_regression",
}


def register_experiment(name: str):
    """Decorator to register an experiment class under a unique name for later instantiation."""

    def decorator(cls: type[BaseExperiment]) -> type[BaseExperiment]:
        """Registers the decorated experiment class in the global registry under the specified name."""
        _REGISTRY[name] = cls
        return cls

    return decorator


class ExperimentFactory:
    """Factory class for creating experiment instances by name from the registry."""

    @staticmethod
    def build(name: str, config) -> BaseExperiment:
        """Creates and returns an instance of the experiment class registered under the given name."""
        resolved_name = _ALIASES.get(name, name)
        if resolved_name not in _REGISTRY:
            available = ", ".join(sorted(_REGISTRY))
            raise ValueError(f"Nieznany eksperyment: {name}. Dostepne: {available}")
        return _REGISTRY[resolved_name](config)

    @staticmethod
    def list() -> list[str]:
        """Returns a sorted list of all registered experiment names."""
        return sorted(_REGISTRY)
