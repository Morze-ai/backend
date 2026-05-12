"""CLI command that compares multiple experiments, exports tabular/JSON reports, and renders comparison bar charts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import ComparisonConfig, ProjectConfig
from src.utils.io import ensure_parent, read_evaluation_report, write_json


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the compare-experiments command."""
    parser = argparse.ArgumentParser(description="Compare multiple experiment outputs.")
    parser.add_argument("config", help="Path to comparison YAML config.")
    return parser


def _collect_row(path: str) -> dict[str, Any]:
    """Collect a row of comparison data from an experiment path, config, or evaluation JSON."""
    path_obj = Path(path)

    # 1. If it's directly an evaluation JSON, use it.
    if path_obj.suffix == ".json" and path_obj.name == "evaluation.json":
        row = read_evaluation_report(path_obj)
        summary_path = path_obj.with_name("training_summary.json")
        row["training_summary_json"] = str(summary_path) if summary_path.exists() else "n/a"
        return row

    # 2. If it's a directory, look for evaluation.json inside.
    if path_obj.is_dir():
        eval_path = path_obj / "evaluation.json"
        if eval_path.exists():
            row = read_evaluation_report(eval_path)
            summary_path = eval_path.with_name("training_summary.json")
            row["training_summary_json"] = str(summary_path) if summary_path.exists() else "n/a"
            return row

    # 3. If it's a YAML config, load it to find where the artifacts should be.
    # But as per "source of truth", we look into reports/ based on config if needed.
    if path_obj.suffix in (".yaml", ".yml"):
        project = ProjectConfig.from_yaml(path_obj)
        eval_path = project.paths.evaluation_json
        if eval_path.exists():
            row = read_evaluation_report(eval_path)
            summary_path = project.paths.training_summary_json
            row["training_summary_json"] = str(summary_path) if summary_path.exists() else "n/a"
            return row

    # Fallback/Error: if we can't find the artifact, we might have to raise or return dummy.
    # For now, let's try to be helpful and return a row with unknown values if path doesn't exist.
    return {
        "experiment_name": path_obj.stem,
        "model_name": "missing",
        "task_type": "unknown",
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "brier_score": 0.0,
        "test_rows": 0,
        "best_validation_accuracy": 0.0,
        "evaluation_json": str(path_obj),
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
        f"- **Best Experiment**: {best['experiment_name']}",
        f"- **Model Type**: {best['model_name']}",
        f"- **Test Accuracy**: {best['accuracy']:.4f}",
        f"- **Test F1 Score**: {best['f1_score']:.4f}",
        "",
        "## Ranking",
        _df_to_markdown_table(
            frame[["rank", "experiment_name", "model_name", "accuracy", "f1_score"]]
        ),
        "",
        "## Detailed Metrics",
        _df_to_markdown_table(
            frame[["experiment_name", "precision", "recall", "brier_score", "test_rows"]]
        ),
        "",
        "## Artifacts",
    ]

    for _, row in frame.iterrows():
        lines.append(f"- **{row['experiment_name']}**:")
        lines.append(f"  - Evaluation: `{row['evaluation_json']}`")
        if row.get("training_summary_json") and row["training_summary_json"] != "n/a":
            lines.append(f"  - Summary: `{row['training_summary_json']}`")

    return "\n".join(lines)


def command(config_path: str) -> pd.DataFrame:
    """Run the experiment comparison workflow."""
    config = ComparisonConfig.from_yaml(config_path)
    rows = [_collect_row(str(path)) for path in config.experiments]
    frame = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    frame["rank"] = range(1, len(frame) + 1)

    ensure_parent(config.comparison_csv)
    frame.to_csv(config.comparison_csv, index=False)
    write_json(config.comparison_json, {"rows": frame.to_dict(orient="records")})

    print("\nRanking eksperymentów (według accuracy):")
    cols_to_show = ["rank", "experiment_name", "model_name", "accuracy", "f1_score"]
    print(frame[cols_to_show].to_string(index=False))

    best = frame.iloc[0]
    print(f"\nNajlepszy eksperyment: {best['experiment_name']}")
    print(f"  Model: {best['model_name']}")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    print(f"  F1 Score: {best['f1_score']:.4f}")

    if config.comparison_markdown:
        report_md = _generate_markdown_report(frame, config)
        ensure_parent(config.comparison_markdown)
        config.comparison_markdown.write_text(report_md, encoding="utf-8")
        print(f"\nRaport Markdown zapisany w: {config.comparison_markdown}")

    ensure_parent(config.comparison_plot_png)
    fig, axis = plt.subplots(figsize=(10, 5), dpi=160)
    axis.bar(frame["experiment_name"], frame["accuracy"])
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
