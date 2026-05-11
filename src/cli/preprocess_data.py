"""CLI command that loads raw data and runs experiment-specific preprocessing to produce split and normalized artifacts."""

from __future__ import annotations

import argparse

from src.cli import build_experiment, load_raw_frame


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the preprocess-data command."""
    parser = argparse.ArgumentParser(description="Run preprocessing and produce split artifacts.")
    parser.add_argument("config", help="Path to project YAML config.")
    return parser


def command(config_path: str) -> None:
    """Run the preprocessing workflow."""
    config, experiment = build_experiment(config_path)
    frame = load_raw_frame(config)
    experiment.preprocess(frame)


def main() -> None:
    """Parse command-line arguments and run the preprocessing command."""
    args = build_parser().parse_args()
    command(config_path=args.config)


if __name__ == "__main__":
    main()
