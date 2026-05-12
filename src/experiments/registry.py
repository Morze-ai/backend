"""Provides decorator-based experiment registration and factory-based experiment instantiation by name."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from src.experiments.base import BaseExperiment

ExperimentType = TypeVar("ExperimentType", bound=type[BaseExperiment])


@dataclass(frozen=True)
class ExperimentMetadata:
    """Stores metadata about a registered experiment."""

    name: str
    cls: type[BaseExperiment]
    description: str = ""
    model_type: str = ""
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    default_config: str = ""


_REGISTRY: dict[str, ExperimentMetadata] = {}
_ALIAS_TO_NAME: dict[str, str] = {}


def register_experiment(
    name: str,
    description: str = "",
    model_type: str = "",
    aliases: list[str] | None = None,
    tags: list[str] | None = None,
    default_config: str = "",
):
    """Decorator to register an experiment class with its metadata."""
    if aliases is None:
        aliases = []
    if tags is None:
        tags = []

    def decorator(cls: type[BaseExperiment]) -> type[BaseExperiment]:
        """Registers the decorated experiment class and its metadata in the global registry."""
        metadata = ExperimentMetadata(
            name=name,
            cls=cls,
            description=description,
            model_type=model_type,
            aliases=aliases,
            tags=tags,
            default_config=default_config,
        )
        _REGISTRY[name] = metadata
        for alias in aliases:
            if alias in _ALIAS_TO_NAME and _ALIAS_TO_NAME[alias] != name:
                existing = _ALIAS_TO_NAME[alias]
                print(
                    f"Ostrzeżenie: Alias '{alias}' jest już przypisany do '{existing}'. Nadpisywanie dla '{name}'."
                )
            _ALIAS_TO_NAME[alias] = name
        return cls

    return decorator


class ExperimentFactory:
    """Factory class for creating experiment instances and accessing metadata."""

    @staticmethod
    def build(name: str, config: Any) -> BaseExperiment:
        """Creates and returns an instance of the experiment class registered under the given name or alias."""
        resolved_name = _ALIAS_TO_NAME.get(name, name)
        if resolved_name not in _REGISTRY:
            available = ", ".join(sorted(_REGISTRY.keys()))
            raise ValueError(f"Nieznany eksperyment: {name}. Dostepne: {available}")

        metadata = _REGISTRY[resolved_name]
        return metadata.cls(config)

    @staticmethod
    def list() -> list[str]:
        """Returns a sorted list of all registered experiment names."""
        return sorted(_REGISTRY.keys())

    @staticmethod
    def get_metadata(name: str) -> ExperimentMetadata:
        """Returns metadata for the specified experiment name or alias."""
        resolved_name = _ALIAS_TO_NAME.get(name, name)
        if resolved_name not in _REGISTRY:
            raise ValueError(f"Nieznany eksperyment: {name}")
        return _REGISTRY[resolved_name]
