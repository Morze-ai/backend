"""Verifies DataLoader creation, target encoding, and label validation in the training helpers."""

# pyright: reportPrivateImportUsage=false

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.trainer import (
    TrainingBundle,
    dataframe_to_loader,
)


def create_sample_training_bundle(
    n_samples: int = 100,
    n_features: int = 5,
    task_type: str = "multiclass",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Helper to create sample training data and metadata."""
    np.random.seed(42)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    features = np.random.randn(n_samples, n_features).astype(np.float32)

    if task_type == "binary":
        targets = np.random.choice(["low", "high"], n_samples)
        class_names = ["low", "high"]
    else:
        targets = np.random.choice(["low", "medium", "high"], n_samples)
        class_names = ["low", "medium", "high"]

    df = pd.DataFrame(features, columns=feature_names)
    df["target"] = targets

    return df, feature_names, class_names


def test_dataframe_to_loader_basic() -> None:
    """Test basic DataLoader creation from DataFrame."""
    df, feature_cols, class_names = create_sample_training_bundle(n_samples=32, n_features=4)

    loader = dataframe_to_loader(
        frame=df,
        feature_columns=feature_cols,
        class_names=class_names,
        target_column="target",
        batch_size=8,
        shuffle=False,
        task_type="multiclass",
    )

    assert loader is not None
    assert len(loader) > 0

    # Check first batch
    features, labels = next(iter(loader))
    assert features.shape[0] == 8  # batch_size
    assert features.shape[1] == len(feature_cols)
    assert labels.shape[0] == 8


def test_dataframe_to_loader_binary_task() -> None:
    """Test DataLoader creation for binary classification."""
    df, feature_cols, class_names = create_sample_training_bundle(
        n_samples=32,
        n_features=4,
        task_type="binary",
    )

    loader = dataframe_to_loader(
        frame=df,
        feature_columns=feature_cols,
        class_names=class_names,
        target_column="target",
        batch_size=8,
        shuffle=False,
        task_type="binary",
    )

    features, labels = next(iter(loader))
    assert features.dtype == torch.float32
    assert labels.dtype == torch.float32


def test_dataframe_to_loader_multiclass_task() -> None:
    """Test DataLoader creation for multiclass classification."""
    df, feature_cols, class_names = create_sample_training_bundle(
        n_samples=32,
        n_features=4,
        task_type="multiclass",
    )

    loader = dataframe_to_loader(
        frame=df,
        feature_columns=feature_cols,
        class_names=class_names,
        target_column="target",
        batch_size=8,
        shuffle=False,
        task_type="multiclass",
    )

    features, labels = next(iter(loader))
    assert features.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_dataframe_to_loader_label_encoding() -> None:
    """Test that labels are correctly encoded to indices."""
    df, feature_cols, class_names = create_sample_training_bundle(n_samples=32, n_features=4)

    loader = dataframe_to_loader(
        frame=df,
        feature_columns=feature_cols,
        class_names=class_names,
        target_column="target",
        batch_size=32,  # Single batch to get all labels
        shuffle=False,
        task_type="multiclass",
    )

    _features, labels = next(iter(loader))

    # Labels should be encoded as indices 0, 1, 2, ...
    unique_labels = torch.unique(labels).numpy()
    assert all(label in range(len(class_names)) for label in unique_labels)


def test_dataframe_to_loader_with_shuffle() -> None:
    """Test that shuffle parameter works."""
    df, feature_cols, class_names = create_sample_training_bundle(n_samples=100, n_features=4)

    # Create two loaders with same data but different shuffle settings
    loader_shuffle = dataframe_to_loader(
        frame=df,
        feature_columns=feature_cols,
        class_names=class_names,
        target_column="target",
        batch_size=10,
        shuffle=True,
        task_type="multiclass",
    )

    loader_no_shuffle = dataframe_to_loader(
        frame=df,
        feature_columns=feature_cols,
        class_names=class_names,
        target_column="target",
        batch_size=10,
        shuffle=False,
        task_type="multiclass",
    )

    # Get first batches
    batch_shuffle = next(iter(loader_shuffle))
    batch_no_shuffle = next(iter(loader_no_shuffle))

    # They should have the same shape
    assert batch_shuffle[0].shape == batch_no_shuffle[0].shape


def test_dataframe_to_loader_unknown_labels() -> None:
    """Test that unknown labels raise appropriate error."""
    df = pd.DataFrame(
        {
            "feature_1": np.random.randn(32),
            "feature_2": np.random.randn(32),
            "target": ["low", "high", "unknown"] * 10
            + ["low", "high"],  # "unknown" not in class_names
        }
    )

    with pytest.raises(ValueError, match="Unknown class labels"):
        dataframe_to_loader(
            frame=df,
            feature_columns=["feature_1", "feature_2"],
            class_names=["low", "high"],
            target_column="target",
            batch_size=8,
            shuffle=False,
            task_type="multiclass",
        )


def test_dataframe_to_loader_missing_target_column() -> None:
    """Test that missing target column raises error."""
    df = pd.DataFrame(
        {
            "feature_1": np.random.randn(32),
            "feature_2": np.random.randn(32),
        }
    )

    with pytest.raises(KeyError):
        dataframe_to_loader(
            frame=df,
            feature_columns=["feature_1", "feature_2"],
            class_names=["low", "high"],
            target_column="nonexistent_target",
            batch_size=8,
            shuffle=False,
            task_type="multiclass",
        )


def test_dataframe_to_loader_missing_feature_column() -> None:
    """Test that missing feature column raises error."""
    df = pd.DataFrame(
        {
            "feature_1": np.random.randn(32),
            "target": ["low", "high"] * 16,
        }
    )

    with pytest.raises(KeyError):
        dataframe_to_loader(
            frame=df,
            feature_columns=["feature_1", "nonexistent_feature"],
            class_names=["low", "high"],
            target_column="target",
            batch_size=8,
            shuffle=False,
            task_type="multiclass",
        )


def test_training_bundle_creation() -> None:
    """Test TrainingBundle dataclass creation."""
    train_df = pd.DataFrame(
        {
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
            "target": ["a", "b"] * 25,
        }
    )

    val_df = pd.DataFrame(
        {
            "f1": np.random.randn(20),
            "f2": np.random.randn(20),
            "target": ["a", "b"] * 10,
        }
    )

    bundle = TrainingBundle(
        train_frame=train_df,
        validation_frame=val_df,
        feature_columns=["f1", "f2"],
        class_names=["a", "b"],
        target_column="target",
        task_type="binary",
    )

    assert bundle.train_frame is train_df
    assert bundle.validation_frame is val_df
    assert bundle.feature_columns == ["f1", "f2"]
    assert bundle.class_names == ["a", "b"]
    assert bundle.task_type == "binary"
