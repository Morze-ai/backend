"""Utility helpers for explainability workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    """
    Creates a directory if it does not exist.
    """

    resolved = Path(path)

    resolved.mkdir(parents=True, exist_ok=True)

    return resolved


def dataframe_from_shap_values(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Converts SHAP values into a pandas DataFrame.
    """

    return pd.DataFrame(
        shap_values,
        columns=feature_names,
    )


def mean_absolute_shap_importance(
    shap_values: np.ndarray,
) -> np.ndarray:
    """
    Computes mean absolute SHAP importance.
    """

    return np.abs(shap_values).mean(axis=0)
