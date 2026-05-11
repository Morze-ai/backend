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
        return LinearClassifier(
            input_dim=len(self.config.data.feature_columns),
            num_classes=len(self.config.data.class_names),
        )
