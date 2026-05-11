"""CLI command that generates raw-data exploratory plots and training-history visualizations."""

from __future__ import annotations

import argparse

from src.cli import build_experiment, load_raw_frame
from src.visualization.plots import save_feature_histograms, save_pairplot


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the visualize command."""
    parser = argparse.ArgumentParser(description="Generate experiment visualizations.")
    parser.add_argument("config", help="Path to project YAML config.")
    parser.add_argument(
        "--include-exploratory",
        action="store_true",
        help="Also render pairplot and feature histograms from raw data.",
    )
    return parser


def command(config_path: str, include_exploratory: bool = False) -> None:
    """Run the visualization workflow."""
    config, experiment = build_experiment(config_path)
    experiment.visualize_training()
    if include_exploratory:
        frame = load_raw_frame(config)
        save_pairplot(
            frame=frame,
            feature_columns=config.data.feature_columns,
            target_column=config.data.target_column,
            output_path=config.paths.pairplot_png,
            dpi=config.visualization.figure_dpi,
        )
        save_feature_histograms(
            frame=frame,
            feature_columns=config.data.feature_columns,
            target_column=config.data.target_column,
            output_path=config.paths.feature_hist_png,
            dpi=config.visualization.figure_dpi,
        )


def main() -> None:
    """Parse command-line arguments and run the visualization command."""
    args = build_parser().parse_args()
    command(config_path=args.config, include_exploratory=args.include_exploratory)


if __name__ == "__main__":
    main()
