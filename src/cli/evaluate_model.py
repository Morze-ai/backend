"""CLI command that evaluates a trained checkpoint on test data and writes metrics and predictions."""

from __future__ import annotations

import argparse

from src.cli import build_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the evaluate-model command."""
    parser = argparse.ArgumentParser(description="Evaluate trained model on test split.")
    parser.add_argument("config", help="Path to project YAML config.")
    return parser


def command(config_path: str) -> None:
    """Run the evaluation workflow for a single experiment."""
    _, experiment = build_experiment(config_path)
    experiment.evaluate()


def main() -> None:
    """Parse command-line arguments and run the evaluation command."""
    args = build_parser().parse_args()
    command(config_path=args.config)


if __name__ == "__main__":
    main()
