from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.models.calibration import apply_temperature_scaling, fit_temperature_scaling

# pyright: reportPrivateImportUsage=false


class SimpleBinaryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(np.array([[4.0, -4.0]], dtype=np.float32)))
            self.linear.bias.copy_(torch.from_numpy(np.array([0.0], dtype=np.float32)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs).squeeze(-1)


def test_temperature_scaling_fit_and_apply_binary() -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 0.2, 0.8, 1.0],
            "x2": [1.0, 0.8, 0.2, 0.0],
            "target": ["low", "low", "high", "high"],
        }
    )

    model = SimpleBinaryModel()
    temperature = fit_temperature_scaling(
        model=model,
        frame=frame,
        feature_columns=["x1", "x2"],
        class_names=["low", "high"],
        target_column="target",
        task_type="binary",
    )

    predictions, probabilities = apply_temperature_scaling(
        model=model,
        frame=frame,
        feature_columns=["x1", "x2"],
        class_names=["low", "high"],
        task_type="binary",
        temperature=temperature,
    )

    assert temperature > 0.0
    assert len(predictions) == len(frame)
    assert len(probabilities) == len(frame)
    assert all(len(row) == 2 for row in probabilities)
    assert all(0.0 <= value <= 1.0 for row in probabilities for value in row)
    assert set(predictions).issubset({"low", "high"})
