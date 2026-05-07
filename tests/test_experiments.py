"""Verifies experiment registration, factory lookup, and error handling for unknown experiments."""

from unittest.mock import Mock

import pytest

from src.experiments.registry import _ALIASES, _REGISTRY, ExperimentFactory


def test_experiment_factory_list_available() -> None:
    """Test that ExperimentFactory.list() returns registered experiment names."""
    available_experiments = ExperimentFactory.list()
    assert isinstance(available_experiments, list)
    assert len(available_experiments) > 0
    assert all(isinstance(name, str) for name in available_experiments)
    # Should include the registered experiments
    assert any(name in available_experiments for name in _REGISTRY)


def test_experiment_factory_aliases() -> None:
    """Test that experiment aliases resolve to correct registered names."""
    # Verify aliases map to existing experiments
    for alias, resolved_name in _ALIASES.items():
        assert resolved_name in _REGISTRY, (
            f"Alias '{alias}' points to unregistered experiment '{resolved_name}'"
        )


def test_experiment_factory_build_known_experiment() -> None:
    """Test that building with known experiment names works correctly."""
    mock_config = Mock()
    mock_config.experiment_name = "test_experiment"

    # Test with a registered experiment name
    available = ExperimentFactory.list()
    if available:
        experiment = ExperimentFactory.build(available[0], mock_config)
        assert experiment is not None
        assert hasattr(experiment, "config")


def test_experiment_factory_build_with_alias() -> None:
    """Test that aliases work when building experiments."""
    mock_config = Mock()
    mock_config.experiment_name = "test_experiment"

    # If 'linear' alias exists and maps to 'linear_classifier'
    if "linear" in _ALIASES and _ALIASES["linear"] in _REGISTRY:
        experiment = ExperimentFactory.build("linear", mock_config)
        assert experiment is not None


def test_experiment_factory_unknown_experiment_error() -> None:
    """Test that unknown experiment names raise appropriate error."""
    mock_config = Mock()

    with pytest.raises(ValueError) as exc_info:
        ExperimentFactory.build("nonexistent_experiment", mock_config)

    error_message = str(exc_info.value)
    assert "Nieznany eksperyment" in error_message or "nonexistent_experiment" in error_message


def test_experiment_registry_is_not_empty() -> None:
    """Test that the experiment registry contains at least one registered experiment."""
    assert len(_REGISTRY) > 0, "No experiments registered in ExperimentFactory"
