"""CLI command for single-sample inference with argument validation, feature parsing, and class-probability output."""

from __future__ import annotations

import argparse
import json

from src.cli import build_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the predict command."""
    parser = argparse.ArgumentParser(description="Run single-row inference.")
    parser.add_argument("config", help="Path to project YAML config.")
    parser.add_argument(
        "--values-json",
        required=True,
        help="JSON object with feature values, e.g. '{\"water_level_m\": 0.12}'.",
    )
    return parser


def command(config_path: str, values_json: str) -> tuple[str, list[float]]:
    """Run the prediction workflow."""
    _, experiment = build_experiment(config_path)
    payload = json.loads(values_json)
    if not isinstance(payload, dict):
        raise ValueError("--values-json must decode to an object.")
    normalized_payload = {str(key): float(value) for key, value in payload.items()}
    return experiment.predict_one(normalized_payload)


def main() -> None:
    """Parse command-line arguments and run the prediction command."""
    args = build_parser().parse_args()
    predicted_class, probabilities = command(config_path=args.config, values_json=args.values_json)
    print(
        json.dumps({"predicted_class": predicted_class, "probabilities": probabilities}, indent=2)
    )


if __name__ == "__main__":
    main()
