"""CLI command that generates raw-data exploratory plots and training-history visualizations."""

from __future__ import annotations

import argparse

from src.cli import build_experiment, load_raw_frame
from src.data.feature_engineering import engineer_features
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
    from tqdm import tqdm

    config, experiment = build_experiment(config_path)

    steps = ["Training Curves", "Confusion Matrix"]
    if include_exploratory:
        steps.extend(["Feature Histograms", "Pairplot"])

    pbar = tqdm(total=len(steps), desc=f"Visualizing {config.experiment_name}")

    # Training visualizations
    experiment.visualize_training()
    pbar.update(2)

    if include_exploratory:
        frame = load_raw_frame(config)
        # Engineer features so that sums, deltas etc. are available for plotting
        frame = engineer_features(frame)

        # Filter to columns that actually exist in the frame
        available_features = [col for col in config.data.feature_columns if col in frame.columns]

        if available_features:
            save_feature_histograms(
                frame=frame,
                feature_columns=available_features,
                target_column=config.data.target_column,
                output_path=config.paths.feature_hist_png,
                dpi=config.visualization.figure_dpi,
            )
            pbar.update(1)

            save_pairplot(
                frame=frame,
                feature_columns=available_features,
                target_column=config.data.target_column,
                output_path=config.paths.pairplot_png,
                dpi=config.visualization.figure_dpi,
            )
            pbar.update(1)
        else:
            pbar.update(2)
    pbar.close()


def main() -> None:
    """Parse command-line arguments and run the visualization command."""
    args = build_parser().parse_args()
    command(config_path=args.config, include_exploratory=args.include_exploratory)


if __name__ == "__main__":
    main()
