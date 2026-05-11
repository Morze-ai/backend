"""Generates and saves feature histograms, pairplots, training curves, and confusion matrices with input validation."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from src.utils.io import ensure_parent


def save_feature_histograms(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Saves histograms of the specified feature columns, colored by the target column, with input validation."""
    if not feature_columns:
        raise ValueError("At least one feature column is required to plot histograms.")
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is missing in the provided frame.")

    ensure_parent(output_path)
    columns_per_row = 2
    row_count = math.ceil(len(feature_columns) / columns_per_row)
    fig, axes = plt.subplots(row_count, columns_per_row, figsize=(10, 4 * row_count), dpi=dpi)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, column in zip(axes_flat, feature_columns, strict=False):
        sns.histplot(data=frame, x=column, hue=target_column, kde=True, ax=axis)

    for axis in axes_flat[len(feature_columns) :]:
        axis.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_pairplot(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Saves a pairplot of the specified feature columns, colored by the target column, with input validation."""
    if not feature_columns:
        raise ValueError("At least one feature column is required to plot a pairplot.")
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is missing in the provided frame.")

    ensure_parent(output_path)
    axes = scatter_matrix(
        frame[feature_columns],
        figsize=(10, 10),
        diagonal="hist",
        c=frame[target_column].astype(str).astype("category").cat.codes,
    )
    for row in axes:
        for axis in row:
            axis.set_xlabel(axis.get_xlabel(), fontsize=8)
            axis.set_ylabel(axis.get_ylabel(), fontsize=8)
    plt.gcf().set_dpi(dpi)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_training_curves(history: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Saves training and validation loss and accuracy curves from the training history DataFrame."""
    ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi)
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["validation_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    class_names: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Saves a confusion matrix plot from the true and predicted labels with input validation."""
    ensure_parent(output_path)
    fig, axis = plt.subplots(figsize=(6, 6), dpi=dpi)
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        labels=class_names,
        ax=axis,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
