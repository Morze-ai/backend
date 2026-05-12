"""CLI command that aggregates experiment results, ranks runs, prints top performers, and saves summary CSV output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_parent, read_json


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the report-summary command."""
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation JSON files into one summary."
    )
    parser.add_argument(
        "--reports-root",
        default="reports",
        help="Root directory to scan for evaluation JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/global/experiment_summary.csv",
        help="Output CSV path for aggregated summary.",
    )
    return parser


def command(reports_root: str, output_csv: str) -> pd.DataFrame:
    """Aggregate evaluation reports from the specified root directory and save a summary CSV."""
    root = Path(reports_root)
    rows: list[dict[str, object]] = []
    for json_path in root.rglob("evaluation.json"):
        payload = read_json(json_path)
        rows.append(
            {
                "experiment_name": str(payload.get("experiment_name", json_path.parent.name)),
                "model_name": str(payload.get("model_name", "unknown")),
                "task_type": str(payload.get("task_type", "unknown")),
                "accuracy": float(payload.get("accuracy", 0.0)),
                "precision": float(payload.get("precision", 0.0)),
                "recall": float(payload.get("recall", 0.0)),
                "f1_score": float(payload.get("f1_score", 0.0)),
                "brier_score": float(payload.get("brier_score", 0.0)),
                "test_rows": int(payload.get("test_rows", 0)),
                "evaluation_json": str(json_path),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("accuracy", ascending=False).reset_index(drop=True)

    output_path = Path(output_csv)
    ensure_parent(output_path)
    frame.to_csv(output_path, index=False)
    return frame


def main() -> None:
    """Parse command-line arguments and run the report summary command."""
    args = build_parser().parse_args()
    frame = command(reports_root=args.reports_root, output_csv=args.output_csv)
    print(
        frame.head(10).to_string(index=False) if not frame.empty else "No evaluation reports found."
    )


if __name__ == "__main__":
    main()
