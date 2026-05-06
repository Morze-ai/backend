#!/usr/bin/env python
"""Normalize raw CSV exports and create processed cleaned datasets.

This script handles two steps:
1. Rewrite raw CSVs into a consistent UTF-8/comma-separated format.
2. Emit processed cleaned datasets for the measurement exports that need
   station-name normalization and hydrological-to-calendar date conversion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preprocessing import (
    DAILY_MEASUREMENT_RENAMES,
    ICE_COVERAGE_RENAMES,
    clean_daily_measurements,
    clean_ice_and_vegetation_measurements,
)
from src.utils.io import (
    build_metadata,
    normalize_text_frame,
    read_csv_safe,
    save_csv_with_metadata,
    write_csv_safe,
    write_metadata_json,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

DAILY_MEASUREMENT_COLUMNS = list(DAILY_MEASUREMENT_RENAMES.keys())
ICE_COVERAGE_COLUMNS = list(ICE_COVERAGE_RENAMES.keys())


def infer_schema(path: Path) -> list[str] | None:
    """Return explicit column names for headerless exports when the format is known."""

    name = path.name.lower()
    if "imgw" in name:
        return DAILY_MEASUREMENT_COLUMNS
    if "ice-and-plant-coverage" in name:
        return ICE_COVERAGE_COLUMNS
    return None


def processed_target_path(path: Path, processed_dir: Path) -> Path | None:
    """Return the output path for processed structured datasets when applicable."""

    name = path.name.lower()
    if "imgw" in name or "ice-and-plant-coverage" in name:
        return processed_dir / f"{path.stem}-cleaned.csv"
    return None


def normalize_raw_file(path: Path, overwrite_raw: bool = True) -> Path:
    """Normalize a raw CSV file in place and emit raw metadata."""

    schema = infer_schema(path)
    artifact = read_csv_safe(path, columns=schema)
    normalized = normalize_text_frame(artifact.frame)

    if overwrite_raw:
        write_csv_safe(normalized, path)

    write_metadata_json(
        path.with_name(f"{path.stem}_metadata.json"),
        build_metadata(
            path,
            normalized,
            encoding="utf-8",
            separator=artifact.separator,
            source=str(path),
            description="Normalized raw CSV export.",
            extras={
                "schema": schema,
                "raw_encoding": artifact.encoding,
                "raw_separator": artifact.separator,
                "had_header": artifact.has_header,
            },
        ),
    )
    return path


def emit_processed_file(path: Path, processed_dir: Path) -> Path | None:
    """Create a processed cleaned dataset for known structured exports."""

    target_path = processed_target_path(path, processed_dir)
    if target_path is None:
        return None

    schema = infer_schema(path)
    artifact = read_csv_safe(path, columns=schema)

    if "imgw" in path.name.lower():
        cleaned = clean_daily_measurements(artifact.frame)
    else:
        cleaned = clean_ice_and_vegetation_measurements(artifact.frame)

    save_csv_with_metadata(
        cleaned,
        target_path,
        source=str(path),
        description="Processed cleaned dataset with normalized station names and calendar dates.",
        extras={
            "source_raw": str(path),
            "raw_encoding": artifact.encoding,
            "raw_separator": artifact.separator,
        },
    )
    return target_path


def iter_csv_files(root: Path) -> list[Path]:
    """Return all CSV files under the raw data directory."""

    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize raw CSV datasets and emit processed outputs."
    )
    parser.add_argument(
        "--no-overwrite-raw", action="store_true", help="Do not rewrite raw CSV files in place."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory containing raw CSV exports.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory where processed outputs should be written.",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = iter_csv_files(args.raw_dir)
    if not raw_files:
        print(f"No CSV files found under {args.raw_dir}")
        return

    for csv_path in raw_files:
        normalize_raw_file(csv_path, overwrite_raw=not args.no_overwrite_raw)
        processed_path = emit_processed_file(csv_path, processed_dir)
        print(f"normalized {csv_path}")
        if processed_path is not None:
            print(f"processed  {processed_path}")


if __name__ == "__main__":
    main()
