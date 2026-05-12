"""Provides file-system and JSON helpers for safe artifact writing and loading."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    xr = None
    HAS_XARRAY = False

_CANDIDATE_ENCODINGS = ("utf-8-sig", "utf-8", "cp1250", "iso-8859-2", "latin1")
_MOJIBAKE_HINTS = re.compile(r"[ÊËÁÀÃÇÐÑÒÓÕÖØÙÚÛÜÝÞ£³¤§]")


@dataclass(frozen=True)
class CsvArtifact:
    """Represents a normalized CSV and the parsing details used to load it."""

    frame: pd.DataFrame
    encoding: str
    separator: str
    has_header: bool


@dataclass(frozen=True)
class NetCdfArtifact:
    """Represents a normalized NetCDF and the parsing details used to load it."""

    frame: pd.DataFrame
    encoding: str = "utf-8"
    format: str = "netcdf"
    has_header: bool = True
    separator: str = ","


def read_text_with_fallback(path: Path) -> tuple[str, str]:
    """Read text using a small set of encodings and return the first successful decode."""

    raw_bytes = Path(path).read_bytes()
    last_error: UnicodeDecodeError | None = None

    for encoding in _CANDIDATE_ENCODINGS:
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError as error:
            last_error = error

    if last_error is not None:
        return raw_bytes.decode("utf-8", errors="replace"), "utf-8-replace"

    return raw_bytes.decode("utf-8", errors="replace"), "utf-8-replace"


def detect_separator(sample_text: str, default: str = ",") -> str:
    """Detect the most likely CSV separator from a text sample."""

    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        counts = {delimiter: sample_text.count(delimiter) for delimiter in (",", ";", "\t", "|")}
        if not any(counts.values()):
            return default
        return max(counts, key=lambda delimiter: counts[delimiter])


def looks_like_header(first_row: list[str]) -> bool:
    """Heuristically determine whether the first row contains a header."""

    if not first_row:
        return False

    header_like_tokens = 0
    for token in first_row:
        token = token.strip()
        if not token:
            continue
        if re.search(r"[A-Za-z_]", token):
            header_like_tokens += 1

    return header_like_tokens >= max(1, len(first_row) // 2)


def repair_mojibake(value: str) -> str:
    """Repair common Central European mojibake patterns such as Wis³a -> Wisła."""

    if not value or value.isascii():
        return value

    normalized = unicodedata.normalize("NFC", value)
    candidates = [normalized]

    if _MOJIBAKE_HINTS.search(normalized):
        for source_encoding, target_encoding in (
            ("latin1", "cp1250"),
            ("latin1", "iso-8859-2"),
            ("cp1252", "cp1250"),
        ):
            try:
                candidate = normalized.encode(source_encoding).decode(target_encoding)
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            candidates.append(unicodedata.normalize("NFC", candidate))

    def score(text: str) -> tuple[int, int, int]:
        """Higher is better: more Polish marks, fewer mojibake marks, shorter text."""
        polish_marks = sum(text.count(char) for char in "ąćęłńóśżźĄĆĘŁŃÓŚŻŹ")
        mojibake_marks = sum(text.count(char) for char in "ÊËÁÀÃÇÐÑÒÓÕÖØÙÚÛÜÝÞ£³¤§")
        return (polish_marks, -mojibake_marks, -len(text))

    return max(candidates, key=score)


def normalize_text_value(value: Any) -> Any:
    """Repair text values while leaving non-strings untouched."""

    if not isinstance(value, str):
        return value
    return repair_mojibake(value)


def normalize_text_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Repair text values in object columns of a DataFrame."""

    normalized = frame.copy()
    for column in normalized.select_dtypes(include=["object", "string"]).columns:
        normalized[column] = normalized[column].map(normalize_text_value)
    return normalized


def read_csv_safe(
    path: Path,
    *,
    columns: list[str] | None = None,
    encoding: str | None = None,
    separator: str | None = None,
) -> CsvArtifact:
    """Load a CSV file with encoding and delimiter fallbacks.

    If `columns` are provided, canonical names are enforced. Headerless exports are
    loaded with `columns` as names; headered exports are read with their header row
    and then renamed to `columns` when widths match.
    """

    path = Path(path)
    text, detected_encoding = (
        read_text_with_fallback(path)
        if encoding is None
        else (path.read_text(encoding=encoding), encoding)
    )
    detected_separator = separator or detect_separator(text[:4096])

    sample_rows = [
        row for row in csv.reader(text.splitlines(), delimiter=detected_separator) if row
    ]
    has_header = looks_like_header(sample_rows[0]) if sample_rows else False

    read_kwargs: dict[str, Any] = {
        "sep": detected_separator,
        "encoding": detected_encoding,
    }

    if columns is not None:
        if has_header:
            read_kwargs["header"] = 0
        else:
            read_kwargs["header"] = None
            read_kwargs["names"] = columns
    elif has_header:
        read_kwargs["header"] = 0
    else:
        read_kwargs["header"] = None

    frame = pd.read_csv(StringIO(text), **read_kwargs)
    if columns is not None and has_header and len(frame.columns) == len(columns):
        frame.columns = columns
    frame = normalize_text_frame(frame)
    return CsvArtifact(
        frame=frame, encoding=detected_encoding, separator=detected_separator, has_header=has_header
    )


def read_netcdf_safe(path: Path) -> NetCdfArtifact:
    """Load a NetCDF file and convert it to a DataFrame.

    Converts multi-dimensional NetCDF data into a flat table by:
    - Extracting 1D data variables directly
    - Stacking multi-dimensional variables along their first axis
    - Including coordinate variables as columns
    """

    if not HAS_XARRAY:
        raise ImportError(
            "xarray is required to read NetCDF files. Install it with: pip install xarray netcdf4"
        )
    if xr is None:
        raise ImportError(
            "xarray is required to read NetCDF files. Install it with: pip install xarray netcdf4"
        )

    path = Path(path)

    # Open NetCDF file
    dataset = xr.open_dataset(path)

    # Convert to DataFrame
    # First, extract all data variables and coordinates
    data_dict: dict[str, Any] = {}

    # Add coordinates as columns
    for coord_name in dataset.coords:
        coord_key = str(coord_name)
        coord_data = dataset.coords[coord_name]
        if coord_data.ndim == 1:
            # Flatten 1D coordinate
            data_dict[coord_key] = coord_data.to_pandas()
        else:
            # Skip multi-dimensional coordinates for now
            pass

    # Add data variables
    for var_name in dataset.data_vars:
        var_key = str(var_name)
        var_data = dataset[var_name]
        if var_data.ndim == 1:
            # Direct 1D variable
            data_dict[var_key] = var_data.to_pandas()
        elif var_data.ndim > 1:
            # Stack multi-dimensional variable along first axis
            try:
                stacked = var_data.stack(z=var_data.dims[1:]).to_pandas()
                data_dict[var_key] = stacked
            except Exception:
                # If stacking fails, skip this variable
                pass

    dataset.close()

    # Create DataFrame
    if not data_dict:
        raise ValueError(f"No data found in NetCDF file: {path}")

    # Find the longest column to align everything
    max_len = max(len(v) if hasattr(v, "__len__") else 1 for v in data_dict.values())

    # Create aligned DataFrame
    frame = pd.DataFrame()
    for col_name, col_data in data_dict.items():
        if hasattr(col_data, "__len__") and len(col_data) == max_len:
            frame[col_name] = col_data
        elif hasattr(col_data, "__len__"):
            # Reindex shorter columns
            frame[col_name] = col_data.reindex(range(max_len))
        else:
            # Scalar value
            frame[col_name] = col_data

    # Normalize text in object columns
    frame = normalize_text_frame(frame)

    return NetCdfArtifact(
        frame=frame,
        encoding="utf-8",
        format="netcdf",
        has_header=True,
    )


def write_csv_safe(
    frame: pd.DataFrame, path: Path, *, index: bool = False, encoding: str = "utf-8"
) -> None:
    """Write a DataFrame to CSV using a normalized UTF-8 encoding."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, encoding=encoding)


def build_metadata(
    path: Path,
    frame: pd.DataFrame,
    *,
    encoding: str,
    separator: str,
    source: str | None = None,
    description: str | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact metadata payload for a normalized dataset."""

    metadata: dict[str, Any] = {
        "source": source or str(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "encoding": encoding,
        "separator": separator,
        "normalized_at": datetime.now(UTC).isoformat(),
    }

    if description is not None:
        metadata["description"] = description

    if extras:
        metadata.update(extras)

    return metadata


def build_metadata_netcdf(
    path: Path,
    frame: pd.DataFrame,
    *,
    source: str | None = None,
    description: str | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact metadata payload for a normalized NetCDF dataset."""

    metadata: dict[str, Any] = {
        "source": source or str(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "encoding": "utf-8",
        "format": "netcdf",
        "normalized_at": datetime.now(UTC).isoformat(),
    }

    if description is not None:
        metadata["description"] = description

    if extras:
        metadata.update(extras)

    return metadata


def read_data_safe(path: Path, *, columns: list[str] | None = None) -> CsvArtifact | NetCdfArtifact:
    """Load a data file (CSV or NetCDF) with automatic format detection.

    Args:
        path (Path): Path to the CSV or NetCDF file
        columns (list[str] | None): Optional column names for CSV files

    Returns:
        CsvArtifact or NetCdfArtifact: Loaded data with metadata

    Raises:
        ValueError: If file format is not supported
        ImportError: If xarray is needed but not installed
    """

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return read_csv_safe(path, columns=columns)
    elif suffix in (".nc", ".netcdf"):
        return read_netcdf_safe(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported formats: .csv, .nc, .netcdf"
        )


def write_metadata_json(path: Path, metadata: dict[str, Any]) -> None:
    """Write metadata as a UTF-8 JSON document."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def save_csv_with_metadata(
    frame: pd.DataFrame,
    csv_path: Path,
    *,
    encoding: str = "utf-8",
    separator: str = ",",
    source: str | None = None,
    description: str | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    """Save a CSV and companion metadata JSON."""

    write_csv_safe(frame, csv_path, encoding=encoding)
    metadata = build_metadata(
        csv_path,
        frame,
        encoding=encoding,
        separator=separator,
        source=source,
        description=description,
        extras=extras,
    )
    write_metadata_json(csv_path.with_name(f"{csv_path.stem}_metadata.json"), metadata)
    return csv_path


def ensure_parent(path: Path) -> None:
    """Ensure the parent directory of the given path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dictionary as a UTF-8 JSON document with pretty formatting.
    If used for metadata, use the `write_metadata_json` function instead to ensure a trailing newline and consistent formatting.
    """
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict[str, Any]:
    """Read a dictionary from a UTF-8 JSON document."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_evaluation_report(path: Path) -> dict[str, Any]:
    """Read an evaluation JSON report and return a standardized dictionary of metrics."""
    payload = read_json(path)
    return {
        "experiment_name": str(payload.get("experiment_name", path.parent.name)),
        "model_name": str(payload.get("model_name", "unknown")),
        "task_type": str(payload.get("task_type", "unknown")),
        "accuracy": float(payload.get("accuracy", payload.get("test_accuracy", 0.0))),
        "precision": float(payload.get("precision", payload.get("test_precision", 0.0))),
        "recall": float(payload.get("recall", payload.get("test_recall", 0.0))),
        "f1_score": float(payload.get("f1_score", payload.get("test_f1", 0.0))),
        "brier_score": float(payload.get("brier_score", payload.get("test_brier", 0.0))),
        "test_rows": int(payload.get("test_rows", 0)),
        "best_validation_accuracy": float(payload.get("best_validation_accuracy", 0.0)),
        "evaluation_json": str(path),
    }


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Convenience wrapper around `read_csv_safe` that returns just the DataFrame."""
    return read_csv_safe(path, **kwargs).frame
