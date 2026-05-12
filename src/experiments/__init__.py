"""Exports experiment classes and registry utilities for standardized experiment construction."""

from src.experiments.base import BaseExperiment
from src.experiments.linear_experiment import LinearClassifierExperiment
from src.experiments.logistic_experiment import LogisticRegressionExperiment
from src.experiments.mlp_experiment import MLPClassifierExperiment
from src.experiments.registry import (
    _ALIAS_TO_NAME,
    _REGISTRY,
    ExperimentFactory,
    register_experiment,
)

__all__ = [
    "_ALIAS_TO_NAME",
    "_REGISTRY",
    "BaseExperiment",
    "ExperimentFactory",
    "LinearClassifierExperiment",
    "LogisticRegressionExperiment",
    "MLPClassifierExperiment",
    "register_experiment",
]
