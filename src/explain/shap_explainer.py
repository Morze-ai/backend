"""SHAP explainability utilities."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.explain.utils import ensure_directory
from src.utils.torch_runtime import get_torch_device


class ShapAnalyzer:
    """
    Unified interface for SHAP explainability.
    """

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
    ) -> None:
        self.model = model
        self.device = get_torch_device()
        self.background_data = background_data

        self.explainer: Any = self._create_explainer()

    def _create_explainer(
        self,
    ) -> shap.Explainer:
        """
        Creates the appropriate SHAP explainer
        based on model type.
        """

        if isinstance(
            self.model,
            (LinearRegression, LogisticRegression),
        ):
            return shap.LinearExplainer(
                self.model,
                self.background_data,
            )

        if isinstance(self.model, torch.nn.Module):
            self.model.to(self.device)
            background: Any = torch.as_tensor(
                self.background_data, dtype=torch.float32, device=self.device
            )

            class ShapTorchWrapper(torch.nn.Module):
                def __init__(self, inner_model: torch.nn.Module) -> None:
                    super().__init__()
                    self.inner_model = inner_model

                def forward(self, inputs: Any) -> torch.Tensor:
                    if isinstance(inputs, np.ndarray) or not isinstance(inputs, torch.Tensor):
                        inputs = torch.as_tensor(inputs, dtype=torch.float32)

                    outputs = self.inner_model(inputs)
                    if outputs.ndim == 1:
                        positive = torch.sigmoid(outputs)
                        return torch.stack([1.0 - positive, positive], dim=1)
                    if outputs.ndim == 2 and outputs.shape[1] == 1:
                        positive = torch.sigmoid(outputs.squeeze(1))
                        return torch.stack([1.0 - positive, positive], dim=1)
                    return outputs

            return shap.GradientExplainer(
                ShapTorchWrapper(self.model),
                background,
            )

        return shap.Explainer(
            self.model,
            self.background_data,
        )

    def compute_shap_values(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Computes SHAP values for input samples.
        """

        # Convert inputs to float32 for torch models to ensure compatibility
        if isinstance(self.model, torch.nn.Module):
            arr_X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            explainer: Any = self.explainer
            result = explainer.shap_values(arr_X)
        else:
            result = self.explainer(X)

        # SHAP explainers may return an Explanation object with `.values`
        # or raw numpy arrays / lists depending on the backend.
        shap_values = getattr(result, "values", result)

        if isinstance(shap_values, list):
            return np.asarray(shap_values[0])

        return np.asarray(shap_values)

    def summary_plot(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        output_path: str | Path,
    ) -> None:
        """
        Generates and saves SHAP summary plot.
        """

        ensure_directory(Path(output_path).parent)

        shap.summary_plot(
            shap_values,
            X,
            show=False,
        )

        plt.tight_layout()

        plt.savefig(output_path)

        plt.close()

    def bar_plot(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        output_path: str | Path,
    ) -> None:
        """
        Generates SHAP feature importance bar plot.
        """

        ensure_directory(Path(output_path).parent)

        shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            show=False,
        )

        plt.tight_layout()

        plt.savefig(output_path)

        plt.close()
