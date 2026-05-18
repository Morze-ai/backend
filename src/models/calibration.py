"""Post-hoc probability calibration helpers."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.utils.io import ensure_parent
from src.utils.logger import get_logger
from src.utils.torch_runtime import get_torch_device, prepare_torch_import

prepare_torch_import()


def _build_feature_tensor(frame: pd.DataFrame, feature_columns: list[str]) -> torch.Tensor:
    return torch.from_numpy(frame[feature_columns].to_numpy(dtype=np.float32, copy=True))


def _encode_targets(
    frame: pd.DataFrame,
    class_names: list[str],
    target_column: str,
    task_type: str,
) -> torch.Tensor:
    mapping = {str(name): index for index, name in enumerate(class_names)}
    encoded = frame[target_column].astype(str).map(mapping)
    if encoded.isna().any():
        unknown_labels = sorted(set(frame[target_column].astype(str)) - set(mapping))
        raise ValueError(
            f"Unknown labels in calibration target '{target_column}': {unknown_labels}"
        )

    if task_type == "binary":
        return torch.from_numpy(encoded.to_numpy(dtype=np.float32, copy=True))

    return torch.from_numpy(encoded.to_numpy(dtype=np.int64, copy=True))


def collect_logits(
    model: nn.Module,
    frame: pd.DataFrame,
    feature_columns: list[str],
    task_type: str,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Collect raw logits for the supplied frame using batched inference."""

    device = get_torch_device()
    logger = get_logger("calibration")
    logger.info(f"Collecting logits for {len(frame)} rows on {device} (batch_size={batch_size})...")

    model = model.to(device)
    model.eval()

    all_logits = []
    features_np = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)

    for i in tqdm(range(0, len(features_np), batch_size), desc="Inference", unit="batch"):
        batch_np = features_np[i : i + batch_size]
        batch_tensor = torch.from_numpy(batch_np).to(device)

        with torch.no_grad():
            logits = model(batch_tensor)
            if task_type == "binary" and logits.ndim > 1:
                logits = logits.squeeze(-1)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            all_logits.append(logits.cpu())

    return torch.cat(all_logits, dim=0)


def fit_temperature_scaling(
    model: nn.Module,
    frame: pd.DataFrame,
    feature_columns: list[str],
    class_names: list[str],
    target_column: str,
    task_type: str,
    max_iter: int = 50,
) -> float:
    """Fit a single temperature parameter on a validation frame."""

    if frame.empty:
        return 1.0

    logits = collect_logits(model, frame, feature_columns, task_type)
    device = logits.device
    targets = _encode_targets(frame, class_names, target_column, task_type).to(device)

    temperature = torch.nn.Parameter(torch.as_tensor(1.0, dtype=torch.float32, device=device))
    criterion: nn.Module = (
        nn.BCEWithLogitsLoss() if task_type == "binary" else nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.LBFGS(
        [temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled_logits = logits / temperature.clamp(min=1e-3)
        if task_type == "binary":
            loss = criterion(scaled_logits.view(-1), targets.float().view(-1))
        else:
            loss = criterion(scaled_logits, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=1e-3).item())


def apply_temperature_scaling(
    model: nn.Module,
    frame: pd.DataFrame,
    feature_columns: list[str],
    class_names: list[str],
    task_type: str,
    temperature: float,
) -> tuple[list[str], list[list[float]]]:
    """Return calibrated predictions and probabilities."""

    logits = collect_logits(model, frame, feature_columns, task_type)
    scaled_logits = logits / max(temperature, 1e-3)

    if task_type == "binary":
        if scaled_logits.ndim > 1:
            scaled_logits = scaled_logits.squeeze(-1)
        positive_probabilities = scaled_logits.sigmoid()
        probabilities = torch.stack(
            [1.0 - positive_probabilities, positive_probabilities], dim=1
        ).tolist()
        prediction_indices = (positive_probabilities >= 0.5).long().tolist()
    else:
        probabilities_tensor = scaled_logits.softmax(dim=1)
        probabilities = probabilities_tensor.tolist()
        prediction_indices = scaled_logits.argmax(dim=1).tolist()

    predictions = [class_names[index] for index in prediction_indices]
    return predictions, probabilities


def save_temperature_scaling(path: str | Path, temperature: float) -> None:
    file_path = Path(path)
    ensure_parent(file_path)
    file_path.write_text(json.dumps({"temperature": float(temperature)}), encoding="utf-8")


def load_temperature_scaling(path: str | Path) -> float:
    file_path = Path(path)
    if not file_path.exists():
        return 1.0

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    value = float(payload.get("temperature", 1.0))
    return value if value > 0 else 1.0
