"""Explainability report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.explain.utils import ensure_directory
from src.explain.visual_report import VisualReportGenerator


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
    from src.events.attribution import attribute_event_type

    output = Path(output_path)
    ensure_directory(output.parent)

    top_features_list = importance_df.head(5)["feature"].tolist()
    primary_factor = attribute_event_type(top_features_list)

    lines: list[str] = []

    lines.append(f"# SHAP Explainability Report - {model_name}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"The primary driver for this model's predictions is: **{primary_factor.value}**.")
    lines.append("")
    lines.append("## Top Features")
    lines.append("")

    top_features = importance_df.head(10)
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


def generate_visual_report(
    importance_df: pd.DataFrame,
    metrics: dict[str, Any],
    output_path: str | Path,
    experiment_name: str,
    plots_dir: str | Path,
) -> None:
    """
    Generates a premium visual PDF report.
    """
    generator = VisualReportGenerator(output_path)
    generator.generate_report(
        experiment_name=experiment_name,
        metrics=metrics,
        importance_df=importance_df,
        plots_dir=Path(plots_dir),
    )
