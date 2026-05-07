"""Definition of the logistic regression model for water level prediction."""

from __future__ import annotations

import torch

from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()


class LogisticRegression(torch.nn.Module):
    """Defines a single-layer logistic regression classifier module for water level prediction."""

    def __init__(self, input_dim: int):
        """Initialization of a binary logistic regression model with one output logit."""
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits for BCEWithLogitsLoss."""
        return self.linear(inputs).squeeze(-1)
