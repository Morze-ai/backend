"""Implements the linear-classifier experiment by binding base workflow logic to the linear model architecture."""

from __future__ import annotations

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.linear import LinearClassifier


@register_experiment("linear_classifier")
class LinearClassifierExperiment(BaseExperiment):
    """Implements the linear classifier experiment by binding base workflow logic to the linear model architecture."""

    @classmethod
    def name(cls) -> str:
        return "linear_classifier"

    def build_model(self) -> LinearClassifier:
        num_classes = (
            1 if len(self.config.data.class_names) == 2 else len(self.config.data.class_names)
        )
        return LinearClassifier(
            input_dim=len(self.config.data.feature_columns),
            num_classes=num_classes,
        )
