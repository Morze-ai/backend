"""Builds and reads datasets, including dataset loading, column normalization, and dataset metadata creation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetArtifact:
    """Represents a dataset artifact, which includes the dataset as a DataFrame and its associated metadata."""

    frame: pd.DataFrame
    metadata: dict[str, Any]


def load_dataset(path: Path, metadata_path: Path | None = None) -> DatasetArtifact:
    """Loads a dataset from the specified path and returns it as a DatasetArtifact.

    Args:
        path (Path): The path to the dataset file.
        metadata_path (Path | None): Optional path to metadata JSON file. If None, tries standard location.

    Returns:
        DatasetArtifact: The loaded dataset as a DatasetArtifact.
    """
    path = Path(path)
    frame = pd.read_csv(path)

    # Try to load metadata
    metadata = {"path": str(path)}

    # If metadata_path not provided, try to find it by convention
    if metadata_path is None:
        # Look for metadata file with same basename
        metadata_candidates = [
            path.parent / f"{path.stem}_metadata.json",
            path.parent / path.name.replace(".csv", "_metadata.json"),
        ]
        for candidate in metadata_candidates:
            if candidate.exists():
                metadata_path = candidate
                break

    # Load metadata if found
    if metadata_path and Path(metadata_path).exists():
        try:
            with Path.open(metadata_path) as f:
                loaded_meta = json.load(f)
                metadata.update(loaded_meta)
        except (OSError, json.JSONDecodeError):
            pass

    return DatasetArtifact(frame=frame, metadata=metadata)
