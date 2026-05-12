"""CLI command that compares multiple experiments, exports tabular/JSON reports, and renders comparison bar charts."""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.cli.report_artifacts import load_evaluation_row
from src.utils.config import ComparisonConfig
from src.utils.io import ensure_parent, write_json


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the compare-experiments command."""
    parser = argparse.ArgumentParser(description="Compare multiple experiment artifacts.")
    parser.add_argument("config", help="Path to comparison YAML config with artifact paths.")
    return parser


def _collect_row(path: str) -> dict[str, Any]:
    """Collect a row of comparison data from a report artifact path."""
    return load_evaluation_row(path)


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
        "## Artifacts & Missing Data",
    ]

    for _, row in frame.iterrows():
        lines.append(f"- **{row['experiment_name']}**:")
        lines.append(f"  - Evaluation: `{row['evaluation_json']}`")
        if row.get("training_summary_json") and row["training_summary_json"] != "n/a":
            lines.append(f"  - Summary: `{row['training_summary_json']}`")
        if row.get("missing_fields"):
            lines.append(f"  - **Missing fields**: {', '.join(row['missing_fields'])}")

    return "\n".join(lines)


def _generate_html_report(frame: pd.DataFrame, config: ComparisonConfig) -> str:
    """Generate a human-readable HTML summary of the comparison."""
    best = frame.iloc[0]

    # Helper for table rows
    def df_to_html_rows(df: pd.DataFrame) -> str:
        rows = []
        for _, r in df.iterrows():
            cells = "".join(
                f"<td>{v:.4f}</td>" if isinstance(v, float) else f"<td>{v}</td>" for v in r
            )
            rows.append(f"<tr>{cells}</tr>")
        return "\n".join(rows)

    def df_to_html_headers(df: pd.DataFrame) -> str:
        headers = "".join(f"<th>{c}</th>" for c in df.columns)
        return f"<tr>{headers}</tr>"

    ranking_cols = ["rank", "experiment_name", "model_name", "accuracy", "f1_score"]
    detailed_cols = ["experiment_name", "precision", "recall", "brier_score", "test_rows"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Comparison Report: {config.project_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        h1, h2 {{ border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .best-card {{ background-color: #e7f3ef; border: 1px solid #d4e9e2; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .artifact-list {{ list-style-type: none; padding: 0; }}
        .artifact-item {{ background: #fdfdfd; border: 1px solid #eee; margin-bottom: 10px; padding: 15px; border-radius: 4px; }}
        .missing-badge {{ background: #fff3cd; color: #856404; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; margin-right: 4px; display: inline-block; }}
    </style>
</head>
<body>
    <h1>Comparison Report: {config.project_name}</h1>

    <div class="best-card">
        <h2>🏆 Best Experiment: {best["experiment_name"]}</h2>
        <p><strong>Model Type:</strong> {best["model_name"]}<br>
        <strong>Test Accuracy:</strong> {best["accuracy"]:.4f}<br>
        <strong>Test F1 Score:</strong> {best["f1_score"]:.4f}</p>
    </div>

    <h2>Ranking</h2>
    <table>
        <thead>{df_to_html_headers(frame[ranking_cols])}</thead>
        <tbody>{df_to_html_rows(frame[ranking_cols])}</tbody>
    </table>

    <h2>Detailed Metrics</h2>
    <table>
        <thead>{df_to_html_headers(frame[detailed_cols])}</thead>
        <tbody>{df_to_html_rows(frame[detailed_cols])}</tbody>
    </table>

    <h2>Artifacts & Missing Data</h2>
    <div class="artifact-list">
"""
    for _, row in frame.iterrows():
        missing_html = ""
        if row.get("missing_fields"):
            badges = "".join(
                f'<span class="missing-badge">{f}</span>' for f in row["missing_fields"]
            )
            missing_html = f"<p><strong>Missing fields:</strong> {badges}</p>"

        summary_link = ""
        if row.get("training_summary_json") and row["training_summary_json"] != "n/a":
            summary_link = f"<li>Summary: <code>{row['training_summary_json']}</code></li>"

        html += f"""
        <div class="artifact-item">
            <h3>{row["experiment_name"]}</h3>
            <ul>
                <li>Evaluation: <code>{row["evaluation_json"]}</code></li>
                {summary_link}
            </ul>
            {missing_html}
        </div>
"""

    html += """
    </div>
</body>
</html>
"""
    return html


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

    if config.comparison_html:
        report_html = _generate_html_report(frame, config)
        ensure_parent(config.comparison_html)
        config.comparison_html.write_text(report_html, encoding="utf-8")
        print(f"\nRaport HTML zapisany w: {config.comparison_html}")

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
