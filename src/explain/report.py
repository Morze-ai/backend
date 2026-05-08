"""Explainability report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.explain.utils import ensure_directory


def save_feature_importance_csv(
    importance_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Saves ranked feature importance as CSV.
    """

    output = Path(output_path)

    ensure_directory(output.parent)

    importance_df.to_csv(
        output,
        index=False,
    )


def generate_markdown_report(
    importance_df: pd.DataFrame,
    output_path: str | Path,
    model_name: str,
) -> None:
    """
    Generates markdown explainability report.
    """

    output = Path(output_path)

    ensure_directory(output.parent)

    top_features = importance_df.head(10)

    lines: list[str] = []

    lines.append(f"# SHAP Explainability Report - {model_name}")
    lines.append("")
    lines.append("## Top 10 Most Important Features")
    lines.append("")

    for i, (_, row) in enumerate(top_features.iterrows(), start=1):
        lines.append(f"{i}. **{row['feature']}** (importance={row['importance']:.6f})")

    lines.append("")
    lines.append("## Full Feature Ranking")
    lines.append("")

    lines.append("| Feature | Importance |")
    lines.append("|---|---|")

    for _, row in importance_df.iterrows():
        lines.append(f"| {row['feature']} | {row['importance']:.6f} |")

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
