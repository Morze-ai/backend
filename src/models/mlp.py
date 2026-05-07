"""Defines a configurable multilayer perceptron with ReLU hidden blocks and a final classification head."""

from __future__ import annotations

import torch
from torch import nn

from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()


class MLP(nn.Module):
    """Defines a configurable multilayer perceptron with ReLU hidden blocks and a final classification head."""

    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int) -> None:
        """initialization of the model"""
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """forward pass of the model"""
        return self.network(inputs)
