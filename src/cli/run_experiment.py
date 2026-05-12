"""CLI command that runs the full end-to-end experiment lifecycle: preprocess, train, evaluate, and visualize."""

from __future__ import annotations

import argparse

from src.cli import analyze, build_experiment, load_raw_frame


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the run-experiment command."""
    parser = argparse.ArgumentParser(description="Run full experiment pipeline.")
    parser.add_argument("config", help="Path to project YAML config.")
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Do not run statistical analysis after experiment.",
    )
    return parser


def command(config_path: str, run_analyze: bool = True) -> None:
    """Run the full experiment workflow."""
    config, experiment = build_experiment(config_path)
    frame = load_raw_frame(config)
    experiment.run(frame)

    if run_analyze:
        print("\n" + "=" * 70)
        print("RUNNING POST-EXPERIMENT ANALYSIS")
        print("=" * 70)
        _summary, md_path, _json_path, risk_assessment = analyze.command(
            input_path=str(config.paths.predictions_csv),
            dataset_name=config.experiment_name,
        )
        print(f"✓ Statistical analysis complete! Report: {md_path}")

        if risk_assessment:
            print("\n" + "!" * 40)
            print("  OCENA RYZYKA (RISK ASSESSMENT)")
            print("!" * 40)
            for line in risk_assessment:
                clean_line = line.replace("### ", "").replace("**", "")
                if clean_line.strip():
                    print(f"  {clean_line}")
            print("!" * 40)


def main() -> None:
    """Parse command-line arguments and run the experiment command."""
    args = build_parser().parse_args()
    command(config_path=args.config, run_analyze=not args.no_analyze)


if __name__ == "__main__":
    main()
