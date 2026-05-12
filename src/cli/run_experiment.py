"""CLI command that runs the full end-to-end experiment lifecycle: preprocess, train, evaluate, and visualize."""

from __future__ import annotations

import argparse

from src.cli import build_experiment, load_raw_frame


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the run-experiment command."""
    parser = argparse.ArgumentParser(description="Run full experiment pipeline.")
    parser.add_argument("config", help="Path to project YAML config.")
    return parser


def command(config_path: str) -> None:
    """Run the full experiment workflow."""
    config, experiment = build_experiment(config_path)
    frame = load_raw_frame(config)
    experiment.run(frame)


def main() -> None:
    """Parse command-line arguments and run the experiment command."""
    args = build_parser().parse_args()
    command(config_path=args.config)


if __name__ == "__main__":
    main()
