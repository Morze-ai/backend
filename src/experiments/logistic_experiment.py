"""Implements the logistic regression experiment, which trains a logistic regression model on the provided training data and evaluates its performance on the validation set."""

from __future__ import annotations

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.logistic_regression import LogisticRegression


@register_experiment("logistic_regression")
class LogisticRegressionExperiment(BaseExperiment):
    @classmethod
    def name(cls) -> str:
        return "logistic_regression"

    def build_model(self) -> LogisticRegression:
        if len(self.config.data.class_names) != 2:
            raise ValueError(
                "Logistic regression supports binary classification only. "
                f"Got {len(self.config.data.class_names)} class names."
            )
        return LogisticRegression(input_dim=len(self.config.data.feature_columns))
