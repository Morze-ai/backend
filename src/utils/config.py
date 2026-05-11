"""Defines typed project and comparison configuration schemas and YAML loading with path normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PathConfig(BaseModel):
    """Defines the paths for raw data, processed data, model checkpoints, and various artifacts used in the project."""

    raw_csv: Path
    dataset_metadata: Path
    processed_dir: Path
    preprocessor_artifact: Path
    model_checkpoint: Path
    training_history_csv: Path
    training_summary_json: Path
    evaluation_json: Path
    predictions_csv: Path
    confusion_matrix_png: Path
    pairplot_png: Path
    feature_hist_png: Path
    training_curves_png: Path


class DataConfig(BaseModel):
    """Defines the data-related configuration, including target and feature columns, class names, and dataset splitting ratios."""

    target_column: str
    feature_columns: list[str]
    class_names: list[str]
    test_size: float = Field(gt=0.0, lt=1.0)
    validation_size: float = Field(gt=0.0, lt=1.0)
    split_strategy: Literal["random", "temporal"] = "random"
    timestamp_column: str = "timestamp"
    validation_start: str | None = None
    test_start: str | None = None


class TrainingConfig(BaseModel):
    """Defines the training-related configuration, including network architecture and optimization parameters."""

    hidden_dims: list[int]
    learning_rate: float = Field(gt=0.0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    weight_decay: float = Field(ge=0.0)


class VisualizationConfig(BaseModel):
    """Defines the visualization-related configuration, including figure DPI and other parameters for plotting."""

    figure_dpi: int = Field(default=160, gt=0)


class PreprocessingConfig(BaseModel):
    """Defines the preprocessing-related configuration, including the normalization strategy to apply to the features."""

    strategy: Literal["zscore", "minmax", "robust"]


class FeatureEngineeringConfig(BaseModel):
    """Defines feature engineering configuration for advanced features like lags and rolling statistics."""

    generate_lag_features: bool = Field(
        default=True,
        description="Generate lag features for weather columns",
    )
    lag_hours: int = Field(
        default=72,
        gt=0,
        description="Number of hours to lag for rainfall, temperature, pressure",
    )
    generate_rolling_features: bool = Field(
        default=False,
        description="Generate rolling window aggregates",
    )
    rolling_windows: list[int] = Field(
        default=[3, 6, 12, 24],
        description="Window sizes in hours for rolling aggregates",
    )
    generate_seasonal_features: bool = Field(
        default=False,
        description="Generate seasonal and temporal features",
    )


class ModelConfig(BaseModel):
    """Defines the model-related configuration, including the type of model to train."""

    name: Literal[
        "mlp_classifier",
        "linear_classifier",
        "logistic_regression",
    ]


class ProjectConfig(BaseModel):
    """Defines the overall project configuration, including paths, data settings, preprocessing, model architecture, training parameters, and visualization settings."""

    model_config = ConfigDict(validate_assignment=True)

    project_name: str
    experiment_name: str
    random_seed: int = 42
    paths: PathConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    model: ModelConfig
    training: TrainingConfig
    visualization: VisualizationConfig

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> ProjectConfig:
        """Loads the project configuration from a YAML file and returns a ProjectConfig instance."""
        path = Path(config_path)
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls(**payload)

    @field_validator("paths", mode="before")
    @classmethod
    def _convert_paths(cls, value: dict[str, Any] | PathConfig) -> dict[str, Path] | PathConfig:
        """Converts string paths in the input dictionary to Path objects for the 'paths' field."""
        if isinstance(value, PathConfig):
            return value
        return {key: Path(raw) for key, raw in value.items()}


class ComparisonConfig(BaseModel):
    """Defines the configuration for comparing multiple experiments, including the list of experiment config paths and output paths for comparison results."""

    project_name: str
    experiments: list[Path]
    comparison_csv: Path
    comparison_json: Path
    comparison_plot_png: Path

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> ComparisonConfig:
        """Loads the comparison configuration from a YAML file and returns a ComparisonConfig instance."""
        path = Path(config_path)
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls(**payload)

    @field_validator("experiments", mode="before")
    @classmethod
    def _convert_experiments(cls, value: list[str]) -> list[Path]:
        """Converts a list of string paths to a list of Path objects for the 'experiments' field."""
        return [Path(raw) for raw in value]
