"""Command-line entry points."""

from __future__ import annotations

from pathlib import Path

import src.experiments  # noqa: F401
from src.data.dataset import load_dataset
from src.experiments.registry import ExperimentFactory
from src.utils.config import ProjectConfig


def load_project_config(config_path: str | Path) -> ProjectConfig:
    """Load and validate project configuration from YAML."""

    return ProjectConfig.from_yaml(config_path)


def build_experiment(config_path: str | Path):
    """Build an experiment instance from a project config file."""

    config = load_project_config(config_path)
    experiment = ExperimentFactory.build(config.model.name, config)
    return config, experiment


def load_raw_frame(config: ProjectConfig):
    """Load the configured raw dataset DataFrame."""

    artifact = load_dataset(config.paths.raw_csv, config.paths.dataset_metadata)
    return artifact.frame
