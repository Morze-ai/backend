"""Verifies YAML configuration loading, path conversion, validation, and expected failure cases."""

from pathlib import Path

import pytest

from src.utils.config import DataConfig, PathConfig, ProjectConfig, TrainingConfig


def test_path_config_creation() -> None:
    """Test that PathConfig correctly initializes with Path objects."""
    path_config = PathConfig(
        raw_csv=Path("data/raw/data.csv"),
        dataset_metadata=Path("data/raw/metadata.json"),
        processed_dir=Path("data/processed"),
        preprocessor_artifact=Path("models/preprocessor.pkl"),
        model_checkpoint=Path("models/checkpoint.pt"),
        training_history_csv=Path("reports/history.csv"),
        training_summary_json=Path("reports/summary.json"),
        evaluation_json=Path("reports/evaluation.json"),
        predictions_csv=Path("reports/predictions.csv"),
        confusion_matrix_png=Path("reports/confusion_matrix.png"),
        pairplot_png=Path("reports/pairplot.png"),
        feature_hist_png=Path("reports/feature_hist.png"),
        training_curves_png=Path("reports/training_curves.png"),
    )
    assert isinstance(path_config.raw_csv, Path)
    assert path_config.raw_csv.name == "data.csv"


def test_data_config_validation() -> None:
    """Test that DataConfig validates test_size and validation_size constraints."""
    # Valid configuration
    valid_config = DataConfig(
        target_column="target",
        feature_columns=["f1", "f2", "f3"],
        class_names=["low", "medium", "high"],
        test_size=0.2,
        validation_size=0.1,
    )
    assert valid_config.test_size == 0.2
    assert valid_config.validation_size == 0.1

    temporal_config = DataConfig(
        target_column="target",
        feature_columns=["f1", "f2"],
        class_names=["low", "high"],
        test_size=0.2,
        validation_size=0.1,
        split_strategy="temporal",
        timestamp_column="timestamp",
        validation_start="2023-10-01",
        test_start="2024-01-01",
    )
    assert temporal_config.split_strategy == "temporal"
    assert temporal_config.validation_start == "2023-10-01"
    assert temporal_config.test_start == "2024-01-01"

    # Invalid test_size (must be between 0 and 1)
    with pytest.raises(ValueError):
        DataConfig(
            target_column="target",
            feature_columns=["f1", "f2"],
            class_names=["low", "high"],
            test_size=1.5,
            validation_size=0.1,
        )

    # Invalid validation_size (must be greater than 0)
    with pytest.raises(ValueError):
        DataConfig(
            target_column="target",
            feature_columns=["f1", "f2"],
            class_names=["low", "high"],
            test_size=0.2,
            validation_size=0.0,
        )


def test_training_config_validation() -> None:
    """Test that TrainingConfig validates positive numeric constraints."""
    # Valid configuration
    valid_config = TrainingConfig(
        hidden_dims=[64, 32],
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        weight_decay=0.0001,
    )
    assert valid_config.learning_rate == 0.001
    assert valid_config.weight_decay == 0.0001

    # Invalid learning_rate (must be positive)
    with pytest.raises(ValueError):
        TrainingConfig(
            hidden_dims=[64, 32],
            learning_rate=-0.001,
            epochs=100,
            batch_size=32,
            weight_decay=0.0001,
        )

    # Invalid epochs (must be positive)
    with pytest.raises(ValueError):
        TrainingConfig(
            hidden_dims=[64, 32],
            learning_rate=0.001,
            epochs=0,
            batch_size=32,
            weight_decay=0.0001,
        )


def test_project_config_from_yaml(tmp_path) -> None:
    """Test that ProjectConfig can load and parse YAML configuration files."""
    config_content = """
project_name: test_project
experiment_name: test_experiment
random_seed: 42

paths:
    raw_csv: data/raw/test.csv
    dataset_metadata: data/raw/metadata.json
    processed_dir: data/processed
    preprocessor_artifact: models/preprocessor.pkl
    model_checkpoint: models/checkpoint.pt
    training_history_csv: reports/history.csv
    training_summary_json: reports/summary.json
    evaluation_json: reports/evaluation.json
    predictions_csv: reports/predictions.csv
    confusion_matrix_png: reports/confusion_matrix.png
    pairplot_png: reports/pairplot.png
    feature_hist_png: reports/feature_hist.png
    training_curves_png: reports/training_curves.png

data:
    target_column: water_level_category
    feature_columns: [water_level_m, flow_m3_s, temperature_c]
    class_names: [low, medium, high]
    test_size: 0.2
    validation_size: 0.1
    split_strategy: temporal
    timestamp_column: timestamp
    validation_start: "2023-10-01"
    test_start: "2024-01-01"

preprocessing:
    strategy: zscore

model:
    name: linear_classifier

training:
    hidden_dims: [64, 32]
    learning_rate: 0.001
    epochs: 100
    batch_size: 32
    weight_decay: 0.0001

visualization:
    figure_dpi: 160
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    config = ProjectConfig.from_yaml(config_file)
    assert config.data.target_column == "water_level_category"
    assert config.model.name == "linear_classifier"
    assert config.training.learning_rate == 0.001
    assert len(config.data.feature_columns) == 3


def test_project_config_from_yaml_invalid_path() -> None:
    """Test that loading from non-existent config raises appropriate error."""
    non_existent = Path("/does/not/exist/config.yaml")
    with pytest.raises(FileNotFoundError):
        ProjectConfig.from_yaml(non_existent)
