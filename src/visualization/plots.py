"""Generates and saves feature histograms, pairplots, training curves, and confusion matrices with input validation."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    sns.set_theme(style="whitegrid")

    # Select a subsets of features if there are too many to avoid cluttered plots
    plot_cols = feature_columns[:12]  # Limit to first 12 features for readability

    columns_per_row = 3
    row_count = math.ceil(len(plot_cols) / columns_per_row)
    fig, axes = plt.subplots(row_count, columns_per_row, figsize=(15, 4 * row_count), dpi=dpi)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, column in zip(axes_flat, plot_cols, strict=False):
        sns.histplot(
            data=frame, x=column, hue=target_column, kde=True, ax=axis, palette="viridis", alpha=0.6
        )
        axis.set_title(f"Distribution of {column}", fontsize=12, pad=10)

    for axis in axes_flat[len(plot_cols) :]:
        axis.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
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
    sns.set_theme(style="ticks")

    # Pairplot is expensive, limit features and rows for better visualization
    plot_cols = feature_columns[:6]

    # Sample rows if necessary
    sample_size = min(1000, len(frame))
    plot_frame = frame.sample(n=sample_size, random_state=42) if len(frame) > 1000 else frame

    # Add target column for hue
    data_to_plot = plot_frame[[*plot_cols, target_column]].copy()

    # Create the pairplot
    g = sns.pairplot(
        data_to_plot,
        hue=target_column,
        palette="husl",
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 30, "edgecolor": "w"},
        height=2.5,
    )

    g.fig.suptitle(f"Feature Pairplot - {target_column}", fontsize=16, y=1.02)
    g.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)


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
