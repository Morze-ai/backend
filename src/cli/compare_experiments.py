"""CLI command that compares multiple experiments, exports tabular/JSON reports, and renders comparison bar charts."""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import ComparisonConfig, ProjectConfig
from src.utils.io import ensure_parent, read_json, write_json


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the compare-experiments command."""
    parser = argparse.ArgumentParser(description="Compare multiple experiment outputs.")
    parser.add_argument("config", help="Path to comparison YAML config.")
    return parser


def _collect_row(config_path: str) -> dict[str, Any]:
    """Collect a row of comparison data from a single experiment config path."""
    project = ProjectConfig.from_yaml(config_path)
    summary = read_json(project.paths.training_summary_json)
    evaluation = read_json(project.paths.evaluation_json)
    return {
        "experiment": project.experiment_name,
        "model": project.model.name,
        "best_validation_accuracy": float(summary.get("best_validation_accuracy", 0.0)),
        "test_accuracy": float(evaluation.get("accuracy", 0.0)),
        "training_summary_json": str(project.paths.training_summary_json),
        "evaluation_json": str(project.paths.evaluation_json),
    }


def command(config_path: str) -> pd.DataFrame:
    """Run the experiment comparison workflow."""
    config = ComparisonConfig.from_yaml(config_path)
    rows = [_collect_row(str(path)) for path in config.experiments]
    frame = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)

    ensure_parent(config.comparison_csv)
    frame.to_csv(config.comparison_csv, index=False)
    write_json(config.comparison_json, {"rows": frame.to_dict(orient="records")})

    ensure_parent(config.comparison_plot_png)
    fig, axis = plt.subplots(figsize=(10, 5), dpi=160)
    axis.bar(frame["experiment"], frame["test_accuracy"])
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Test Accuracy by Experiment")
    axis.set_ylabel("Accuracy")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(config.comparison_plot_png)
    plt.close(fig)
    return frame


def main() -> None:
    """Parse command-line arguments and run the comparison command."""
    args = build_parser().parse_args()
    command(config_path=args.config)


if __name__ == "__main__":
    main()
