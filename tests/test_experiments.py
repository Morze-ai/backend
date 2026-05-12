"""Verifies experiment registration, factory lookup, and metadata contract."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.cli import build_experiment
from src.experiments.registry import _ALIAS_TO_NAME, _REGISTRY, ExperimentFactory


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
    for alias, resolved_name in _ALIAS_TO_NAME.items():
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
    if "linear" in _ALIAS_TO_NAME and _ALIAS_TO_NAME["linear"] in _REGISTRY:
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


def test_registry_metadata_fields_present() -> None:
    """Test that registered experiments expose non-empty metadata fields."""
    for experiment_name in ExperimentFactory.list():
        metadata = ExperimentFactory.get_metadata(experiment_name)
        assert metadata.name == experiment_name
        assert metadata.cls is not None
        assert isinstance(metadata.description, str)
        assert isinstance(metadata.model_type, str)
        assert isinstance(metadata.aliases, list)
        assert isinstance(metadata.tags, list)
        # default_config must be set and point to an existing file
        assert isinstance(metadata.default_config, str)
        assert metadata.default_config != ""
        assert Path(metadata.default_config).exists(), (
            f"default_config for {experiment_name} points to missing file: {metadata.default_config}"
        )


def test_built_experiment_has_config_path() -> None:
    """Test that CLI experiment builder attaches absolute config path for reproducibility metadata."""
    config, experiment = build_experiment("configs/EXAMPLE_linear_minmax.yaml")
    assert config.experiment_name
    assert experiment.config_path is not None
    assert Path(experiment.config_path).is_absolute()


def test_experiment_evaluation_metadata_completeness(tmp_path: Path) -> None:
    """Test that experiment evaluation produces all required metadata fields."""
    config = MagicMock()
    config.experiment_name = "test_exp"
    config.model.name = "test_model"
    config.random_seed = 42
    config.data.target_column = "target"
    config.data.feature_columns = ["feat1"]
    config.data.class_names = ["low", "high"]
    config.paths.evaluation_json = tmp_path / "evaluation.json"
    config.paths.training_summary_json = tmp_path / "summary.json"
    config.paths.predictions_csv = tmp_path / "preds.csv"

    # Mock model_dump for _run_metadata
    config.model_dump.return_value = {
        "project_name": "test_project",
        "experiment_name": "test_exp",
        "random_seed": 42,
        "data": {"target_column": "target"},
    }

    expected_hash = hashlib.sha256(
        json.dumps(config.model_dump.return_value, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()

    # Create dummy processed data
    dummy_data = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            "target": ["low", "high"],
            "feat1": [0.1, 0.2],
        }
    )

    with patch("src.experiments.base.get_logger"):
        from src.experiments.base import BaseExperiment

        class TestExperiment(BaseExperiment):
            @classmethod
            def name(cls) -> str:
                return "test_experiment"

            def build_model(self) -> object:
                return Mock()

        experiment = TestExperiment(config)
        experiment._test_frame = dummy_data
        experiment._best_validation_accuracy = 0.9

        # Mock dependencies of evaluate
        with (
            patch("src.experiments.base.apply_temperature_scaling") as mock_predict,
            patch("src.experiments.base.write_json") as mock_write,
            patch.object(experiment, "load_checkpoint"),
            patch.object(experiment, "build_model"),
            patch.object(experiment, "_task_type", return_value="binary"),
        ):
            # 2 rows, first predicted as low, second as high
            mock_predict.return_value = (["low", "high"], [[0.9, 0.1], [0.1, 0.9]])

            experiment.evaluate()

            # Verify write_json was called for evaluation.json
            payload = mock_write.call_args.args[1]

            assert payload["experiment_name"] == "test_exp"
            assert payload["accuracy"] == 1.0
            assert "precision" in payload
            assert "recall" in payload
            assert "f1_score" in payload
            assert payload["best_validation_accuracy"] == 0.9
            assert payload["random_seed"] == 42
            assert payload["config_path"] == ""
            assert "run_timestamp" in payload
            assert payload["config_hash"] == expected_hash
