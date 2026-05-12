"""CLI command that compares multiple experiments, exports tabular/JSON reports, and renders comparison bar charts."""

from __future__ import annotations

import argparse
from pathlib import Path
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


def _collect_row(path: str) -> dict[str, Any]:
    """Collect a row of comparison data from an experiment config path or evaluation JSON."""
    path_obj = Path(path)
    if path_obj.suffix == ".json":
        evaluation = read_json(path_obj)
        # In this flow, we might not have the training summary easily accessible
        # unless it's in the same directory or its path is stored in evaluation.json
        return {
            "experiment": str(evaluation.get("experiment_name", path_obj.parent.name)),
            "model": str(evaluation.get("model_name", "unknown")),
            "task_type": evaluation.get("task_type", "unknown"),
            "best_validation_accuracy": float(evaluation.get("best_validation_accuracy", 0.0)),
            "test_accuracy": float(evaluation.get("accuracy", 0.0)),
            "test_precision": float(evaluation.get("precision", 0.0)),
            "test_recall": float(evaluation.get("recall", 0.0)),
            "test_f1": float(evaluation.get("f1_score", 0.0)),
            "test_brier": float(evaluation.get("brier_score", 0.0)),
            "test_rows": int(evaluation.get("test_rows", 0)),
            "training_summary_json": "n/a",
            "evaluation_json": str(path_obj),
        }

    project = ProjectConfig.from_yaml(path)
    summary = read_json(project.paths.training_summary_json)
    evaluation = read_json(project.paths.evaluation_json)
    return {
        "experiment": project.experiment_name,
        "model": project.model.name,
        "task_type": evaluation.get("task_type", "unknown"),
        "best_validation_accuracy": float(summary.get("best_validation_accuracy", 0.0)),
        "test_accuracy": float(evaluation.get("accuracy", 0.0)),
        "test_precision": float(evaluation.get("precision", 0.0)),
        "test_recall": float(evaluation.get("recall", 0.0)),
        "test_f1": float(evaluation.get("f1_score", 0.0)),
        "test_brier": float(evaluation.get("brier_score", 0.0)),
        "test_rows": int(evaluation.get("test_rows", 0)),
        "training_summary_json": str(project.paths.training_summary_json),
        "evaluation_json": str(project.paths.evaluation_json),
    }


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Simple helper to convert a DataFrame to a Markdown table without tabulate."""
    if df.empty:
        return ""

    headers = df.columns.tolist()
    header_row = "| " + " | ".join(map(str, headers)) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    body_rows = []
    for _, row in df.iterrows():
        body_row = (
            "| "
            + " | ".join(map(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x), row))
            + " |"
        )
        body_rows.append(body_row)

    return "\n".join([header_row, sep_row, *body_rows])


def _generate_markdown_report(frame: pd.DataFrame, config: ComparisonConfig) -> str:
    """Generate a human-readable Markdown summary of the comparison."""
    best = frame.iloc[0]

    lines = [
        f"# Comparison Report: {config.project_name}",
        "",
        "## Summary",
        f"- **Best Experiment**: {best['experiment']}",
        f"- **Model Type**: {best['model']}",
        f"- **Test Accuracy**: {best['test_accuracy']:.4f}",
        f"- **Test F1 Score**: {best['test_f1']:.4f}",
        "",
        "## Ranking",
        _df_to_markdown_table(frame[["rank", "experiment", "model", "test_accuracy", "test_f1"]]),
        "",
        "## Detailed Metrics",
        _df_to_markdown_table(
            frame[["experiment", "test_precision", "test_recall", "test_brier", "test_rows"]]
        ),
        "",
        "## Artifacts",
    ]

    for _, row in frame.iterrows():
        lines.append(f"- **{row['experiment']}**:")
        lines.append(f"  - Evaluation: `{row['evaluation_json']}`")
        if row["training_summary_json"] != "n/a":
            lines.append(f"  - Summary: `{row['training_summary_json']}`")

    return "\n".join(lines)


def command(config_path: str) -> pd.DataFrame:
    """Run the experiment comparison workflow."""
    config = ComparisonConfig.from_yaml(config_path)
    rows = [_collect_row(str(path)) for path in config.experiments]
    frame = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    frame["rank"] = range(1, len(frame) + 1)

    ensure_parent(config.comparison_csv)
    frame.to_csv(config.comparison_csv, index=False)
    write_json(config.comparison_json, {"rows": frame.to_dict(orient="records")})

    print("\nRanking eksperymentów (według test_accuracy):")
    cols_to_show = ["rank", "experiment", "model", "test_accuracy", "test_f1"]
    print(frame[cols_to_show].to_string(index=False))

    best = frame.iloc[0]
    print(f"\nNajlepszy eksperyment: {best['experiment']}")
    print(f"  Model: {best['model']}")
    print(f"  Accuracy: {best['test_accuracy']:.4f}")
    print(f"  F1 Score: {best['test_f1']:.4f}")

    if config.comparison_markdown:
        report_md = _generate_markdown_report(frame, config)
        ensure_parent(config.comparison_markdown)
        config.comparison_markdown.write_text(report_md, encoding="utf-8")
        print(f"\nRaport Markdown zapisany w: {config.comparison_markdown}")

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
