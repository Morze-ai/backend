"""Defines a single-layer linear classifier module for multiclass Iris prediction."""

import torch
from torch import nn

from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()


class LinearClassifier(nn.Module):
    """Defines a single-layer linear classifier module for multiclass Iris prediction."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        """initialization of the model"""
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """forward pass of the model"""
        return self.linear(inputs)
