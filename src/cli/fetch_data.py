"""CLI command that acquires raw data (local CSV or built-in), writes canonical raw CSV, and stores dataset metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cli import load_project_config
from src.utils.io import build_metadata, read_csv_safe, write_csv_safe, write_metadata_json


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the fetch-data command."""
    parser = argparse.ArgumentParser(description="Normalize and store a raw CSV dataset.")
    parser.add_argument("config", help="Path to project YAML config.")
    parser.add_argument(
        "--source-csv",
        dest="source_csv",
        default=None,
        help="Optional source CSV path. Defaults to config.paths.raw_csv.",
    )
    return parser


def command(config_path: str, source_csv: str | None = None) -> Path:
    """Run the data fetching and normalization workflow."""
    config = load_project_config(config_path)
    source_path = Path(source_csv) if source_csv else config.paths.raw_csv
    artifact = read_csv_safe(source_path)

    write_csv_safe(artifact.frame, config.paths.raw_csv)
    metadata = build_metadata(
        path=config.paths.raw_csv,
        frame=artifact.frame,
        encoding=artifact.encoding,
        separator=artifact.separator,
        source=str(source_path),
        extras={"has_header": artifact.has_header},
    )
    write_metadata_json(config.paths.dataset_metadata, metadata)
    return config.paths.raw_csv


def main() -> None:
    """Parse command-line arguments and run the data fetching command."""
    args = build_parser().parse_args()
    command(config_path=args.config, source_csv=args.source_csv)


if __name__ == "__main__":
    main()
