"""Implements DataFrame-to-DataLoader conversion, epoch training loop, checkpoint selection, and batch inference helpers."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import from_numpy, nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.io import ensure_parent
from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()


@dataclass(frozen=True)
class TrainingBundle:
    """Encapsulates split DataFrames and metadata required by training."""

    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    feature_columns: list[str]
    class_names: list[str]
    target_column: str
    task_type: Literal["multiclass", "binary"] = "multiclass"


@dataclass(frozen=True)
class TrainingResult:
    """Encapsulates training outputs and best validation metrics."""

    model: nn.Module
    history: pd.DataFrame
    best_validation_accuracy: float


def dataframe_to_loader(
    frame: pd.DataFrame,
    feature_columns: list[str],
    class_names: list[str],
    target_column: str,
    batch_size: int,
    shuffle: bool,
    task_type: Literal["multiclass", "binary"] = "multiclass",
) -> DataLoader:
    """Convert a DataFrame into a torch DataLoader with encoded labels."""

    mapping = {str(name): index for index, name in enumerate(class_names)}
    features = from_numpy(frame[feature_columns].to_numpy(dtype=np.float32, copy=True)).clone()
    encoded_labels = frame[target_column].astype(str).map(mapping)
    if encoded_labels.isna().any():
        unknown_labels = sorted(set(frame[target_column].astype(str)) - set(mapping))
        raise ValueError(
            "Unknown class labels found in column "
            f"'{target_column}': {unknown_labels}. Expected one of {sorted(mapping)}"
        )

    if task_type == "binary":
        labels = from_numpy(encoded_labels.to_numpy(dtype=np.float32, copy=True)).clone()
    else:
        labels = from_numpy(encoded_labels.to_numpy(dtype=np.int64, copy=True)).clone()

    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    bundle: TrainingBundle,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    checkpoint_path: Path,
) -> TrainingResult:
    """Train a model and retain best checkpoint by validation accuracy."""

    train_loader = dataframe_to_loader(
        bundle.train_frame,
        bundle.feature_columns,
        bundle.class_names,
        bundle.target_column,
        batch_size=batch_size,
        shuffle=True,
        task_type=bundle.task_type,
    )
    validation_loader = dataframe_to_loader(
        bundle.validation_frame,
        bundle.feature_columns,
        bundle.class_names,
        bundle.target_column,
        batch_size=batch_size,
        shuffle=False,
        task_type=bundle.task_type,
    )

    criterion: nn.Module = (
        nn.BCEWithLogitsLoss() if bundle.task_type == "binary" else nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    ensure_parent(checkpoint_path)
    best_state = model.state_dict()
    best_validation_accuracy = 0.0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_predictions: list[int] = []
        train_targets: list[int] = []

        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            if bundle.task_type == "binary":
                logits = logits.squeeze(-1)
                labels = labels.float()
                loss = criterion(logits, labels)
                probabilities = logits.sigmoid()
                train_predictions.extend((probabilities >= 0.5).long().tolist())
                train_targets.extend(labels.long().tolist())
            else:
                loss = criterion(logits, labels)
                train_predictions.extend(logits.argmax(dim=1).tolist())
                train_targets.extend(labels.tolist())
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        validation_losses: list[float] = []
        validation_predictions: list[int] = []
        validation_targets: list[int] = []
        with torch.no_grad():
            for features, labels in validation_loader:
                logits = model(features)
                if bundle.task_type == "binary":
                    logits = logits.squeeze(-1)
                    labels = labels.float()
                    loss = criterion(logits, labels)
                    probabilities = logits.sigmoid()
                    validation_predictions.extend((probabilities >= 0.5).long().tolist())
                    validation_targets.extend(labels.long().tolist())
                else:
                    loss = criterion(logits, labels)
                    validation_predictions.extend(logits.argmax(dim=1).tolist())
                    validation_targets.extend(labels.tolist())
                validation_losses.append(float(loss.item()))

        train_accuracy = accuracy_score(train_targets, train_predictions)
        validation_accuracy = accuracy_score(validation_targets, validation_predictions)
        mean_train_loss = float(np.mean(train_losses))
        mean_validation_loss = float(np.mean(validation_losses))

        rows.append(
            {
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "validation_loss": mean_validation_loss,
                "train_accuracy": float(train_accuracy),
                "validation_accuracy": float(validation_accuracy),
            }
        )

        if validation_accuracy >= best_validation_accuracy:
            best_validation_accuracy = float(validation_accuracy)
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)

    return TrainingResult(
        model=model,
        history=pd.DataFrame(rows),
        best_validation_accuracy=best_validation_accuracy,
    )


def predict_with_model(
    model: nn.Module,
    frame: pd.DataFrame,
    feature_columns: list[str],
    class_names: list[str],
    task_type: Literal["multiclass", "binary"] = "multiclass",
) -> tuple[list[str], list[list[float]]]:
    """Run model inference and return labels with probabilities."""

    features = from_numpy(frame[feature_columns].to_numpy(dtype=np.float32, copy=True)).clone()
    model.eval()
    with torch.no_grad():
        logits = model(features)
        if task_type == "binary":
            logits = logits.squeeze(-1)
            positive_probabilities = logits.sigmoid()
            prediction_indices = cast(list[int], (positive_probabilities >= 0.5).long().tolist())
            probabilities = [[1.0 - float(prob), float(prob)] for prob in positive_probabilities]
        else:
            probabilities_tensor = logits.softmax(dim=1)
            prediction_indices = cast(list[int], logits.argmax(dim=1).tolist())
            probabilities = cast(list[list[float]], probabilities_tensor.tolist())
    predictions = [class_names[index] for index in prediction_indices]
    return predictions, probabilities
