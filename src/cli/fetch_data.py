"""CLI command that acquires raw data (CSV or NetCDF), normalizes it, and stores dataset metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cli import load_project_config
from src.utils.io import (
    build_metadata,
    build_metadata_netcdf,
    read_data_safe,
    write_csv_safe,
    write_metadata_json,
)


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the fetch-data command."""
    parser = argparse.ArgumentParser(
        description="Normalize and store a raw dataset (CSV or NetCDF)."
    )
    parser.add_argument("config", help="Path to project YAML config.")
    parser.add_argument(
        "--source-data",
        dest="source_data",
        default=None,
        help="Optional source data path (CSV or NetCDF). Defaults to config.paths.raw_csv.",
    )
    return parser


def command(config_path: str, source_data: str | None = None) -> Path:
    """Run the data fetching and normalization workflow."""
    config = load_project_config(config_path)
    source_path = Path(source_data) if source_data else config.paths.raw_csv

    # Load data (auto-detects CSV or NetCDF format)
    artifact = read_data_safe(source_path)

    write_csv_safe(artifact.frame, config.paths.raw_csv)

    # Build metadata based on source format
    if source_path.suffix.lower() in (".nc", ".netcdf"):
        metadata = build_metadata_netcdf(
            path=config.paths.raw_csv,
            frame=artifact.frame,
            source=str(source_path),
        )
    else:
        # artifact is CsvArtifact - safe to access .encoding and .separator
        metadata = build_metadata(
            path=config.paths.raw_csv,
            frame=artifact.frame,
            encoding=artifact.encoding,  # type: ignore
            separator=artifact.separator,  # type: ignore
            source=str(source_path),
            extras={"has_header": artifact.has_header},  # type: ignore
        )

    write_metadata_json(config.paths.dataset_metadata, metadata)
    return config.paths.raw_csv


def main() -> None:
    """Parse command-line arguments and run the data fetching command."""
    args = build_parser().parse_args()
    command(config_path=args.config, source_data=args.source_data)


if __name__ == "__main__":
    main()
