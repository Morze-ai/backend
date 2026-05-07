"""CLI command that builds the selected experiment and executes model training from preprocessed artifacts."""

from __future__ import annotations

import argparse

from src.cli import build_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the train-model command."""
    parser = argparse.ArgumentParser(description="Train model using processed split artifacts.")
    parser.add_argument("config", help="Path to project YAML config.")
    return parser


def command(config_path: str) -> None:
    """Run the model training workflow."""
    _, experiment = build_experiment(config_path)
    experiment.train()


def main() -> None:
    """Parse command-line arguments and run the training command."""
    args = build_parser().parse_args()
    command(config_path=args.config)


if __name__ == "__main__":
    main()
