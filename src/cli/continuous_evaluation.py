"""CLI command for one-shot continuous flood-risk inference."""

from __future__ import annotations

import argparse
import json

from src.continuous.service import build_service


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for continuous evaluation command."""

    parser = argparse.ArgumentParser(description="Run one continuous flood-risk evaluation.")
    parser.add_argument("config", help="Path to project YAML config.")
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Disable writing latest/history continuous evaluation artifacts.",
    )
    return parser


def command(config_path: str, persist: bool = True) -> dict[str, object]:
    """Run one continuous evaluation cycle and return serialized output."""

    service = build_service(config_path)
    result = service.evaluate(persist=persist)
    return result.model_dump(mode="json")


def main() -> None:
    """Entrypoint for command-line execution."""

    args = build_parser().parse_args()
    payload = command(config_path=args.config, persist=not args.no_persist)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
