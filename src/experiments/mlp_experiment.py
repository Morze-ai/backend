"""Implements the configurable deep-MLP experiment using hidden-layer dimensions from configuration."""

from __future__ import annotations

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.mlp import MLP


@register_experiment("mlp_classifier")
class MLPClassifierExperiment(BaseExperiment):
    """Implements the MLP classifier experiment by binding base workflow logic to the MLP model architecture."""

    @classmethod
    def name(cls) -> str:
        return "mlp_classifier"

    def build_model(self) -> MLP:
        num_classes = (
            1 if len(self.config.data.class_names) == 2 else len(self.config.data.class_names)
        )
        return MLP(
            input_dim=len(self.config.data.feature_columns),
            hidden_dims=self.config.training.hidden_dims,
            num_classes=num_classes,
        )
