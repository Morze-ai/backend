"""CLI command that aggregates experiment results, ranks runs, prints top performers, and saves summary CSV output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_parent, read_evaluation_report


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
        rows.append(read_evaluation_report(json_path))

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
