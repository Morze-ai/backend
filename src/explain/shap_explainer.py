"""SHAP explainability utilities."""

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
        self.background_data = background_data

        self.explainer = self._create_explainer()

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
            # Use a numpy float32 array as background to avoid accessing
            # torch.* attributes that some type checkers treat as private.
            background = np.asarray(self.background_data, dtype=np.float32)

            return shap.DeepExplainer(
                self.model,
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

        # Call the explainer using the newer SHAP API which returns an
        # Explanation object; extract `.values` when present. Convert
        # inputs to float32 numpy arrays for torch models to avoid using
        # torch.tensor or torch.float32 attributes directly (type-checker friendly).
        if isinstance(self.model, torch.nn.Module):
            arr_X = np.asarray(X, dtype=np.float32)
            result = self.explainer(arr_X)
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
