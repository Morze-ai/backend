"""CLI command for comprehensive statistical analysis of water level predictions and lag correlations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.statistical_analyzer import StatisticalAnalyzer
from src.events.rules import EVENT_RULES, EventType
from src.utils.io import ensure_parent, read_csv, read_json


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the analyze command."""
    parser = argparse.ArgumentParser(
        description="Perform comprehensive statistical analysis on predictions or evaluation data."
    )
    parser.add_argument(
        "input",
        help="Path to predictions CSV or evaluation.json file",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Output path for markdown statistical report (default: input_dir/statistical_analysis.md)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output path for JSON statistical summary (default: input_dir/statistical_analysis.json)",
    )
    parser.add_argument(
        "--target-column",
        default="water_level_m",
        help="Name of target variable column (default: water_level_m)",
    )
    parser.add_argument(
        "--event-column",
        default="event_occurred",
        help="Name of event binary column (default: event_occurred)",
    )
    parser.add_argument(
        "--soil-saturation-column",
        default="soil_saturation_index",
        help="Name of soil saturation column (default: soil_saturation_index)",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile cutoff for 'high water' definition (default: 95)",
    )
    parser.add_argument(
        "--dataset-name",
        default="unnamed",
        help="Name for reporting purposes (default: unnamed)",
    )
    return parser


def generate_markdown_report(
    summary: Any,
    output_path: Path,
) -> list[str]:
    """Generate markdown report from statistical summary. Returns risk assessment lines."""
    ensure_parent(output_path)

    lines: list[str] = []

    # Header
    lines.append(f"# Statistical Analysis Report - {summary.dataset_name}")
    lines.append("")
    lines.append(f"**Analysis Date**: {pd.Timestamp.now().isoformat()}")
    lines.append("")

    # Dataset info
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(f"- **Total Rows**: {summary.n_total_rows}")
    lines.append(f"- **Time Range**: {summary.timestamp_range[0]} to {summary.timestamp_range[1]}")
    lines.append("")

    # Warnings
    if summary.warnings:
        lines.append("## ⚠️ Warnings")
        lines.append("")
        for warning in summary.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    # Lag Correlations
    if summary.lag_correlations:
        lines.append("## Lag Feature Correlations")
        lines.append("")
        lines.append("Analysis of correlations between lag features and water level target.")
        lines.append("")

        # Group by season
        seasons = sorted(set(lc.season for lc in summary.lag_correlations))
        for season in seasons:
            season_corr = [lc for lc in summary.lag_correlations if lc.season == season]
            if not season_corr:
                continue

            lines.append(f"### Season: {season.upper()}")
            lines.append("")
            lines.append(
                "| Lag (hours) | Feature | Pearson r | Pearson p-value | Spearman rho | Spearman p-value | N Samples |"
            )
            lines.append("|---|---|---|---|---|---|---|")

            for corr in sorted(season_corr, key=lambda x: x.lag_hours):
                pearson_p_str = (
                    f"{corr.pearson_p_value:.4f}" if not pd.isna(corr.pearson_p_value) else "N/A"
                )
                spearman_p_str = (
                    f"{corr.spearman_p_value:.4f}" if not pd.isna(corr.spearman_p_value) else "N/A"
                )
                pearson_r_str = f"{corr.pearson_r:.4f}" if not pd.isna(corr.pearson_r) else "N/A"
                spearman_rho_str = (
                    f"{corr.spearman_rho:.4f}" if not pd.isna(corr.spearman_rho) else "N/A"
                )

                lines.append(
                    f"| {corr.lag_hours} | {corr.feature_name} | {pearson_r_str} | {pearson_p_str} | "
                    f"{spearman_rho_str} | {spearman_p_str} | {corr.n_samples} |"
                )

            lines.append("")

    # Hypothesis Tests
    if summary.hypothesis_tests:
        lines.append("## Hypothesis Tests (High vs Low Water)")
        lines.append("")
        lines.append(
            "Comparing feature values between high and low water level groups using t-test and Mann-Whitney U test."
        )
        lines.append("")

        # Group by season
        seasons = sorted(set(ht.season for ht in summary.hypothesis_tests))
        for season in seasons:
            season_tests = [ht for ht in summary.hypothesis_tests if ht.season == season]
            if not season_tests:
                continue

            lines.append(f"### Season: {season.upper()}")
            lines.append("")

            for test in season_tests:
                lines.append(f"#### {test.feature_name}")
                lines.append("")
                lines.append(f"- **Groups**: {test.group1_label} vs {test.group2_label}")
                lines.append(f"- **Threshold Percentile**: {test.threshold_percentile}%")
                lines.append("")

                lines.append("**T-Test (Parametric)**:")
                lines.append(f"- Statistic: {test.ttest_statistic:.6f}")
                lines.append(f"- P-value (uncorrected): {test.ttest_p_value:.6f}")
                lines.append(f"- P-value (Bonferroni): {test.ttest_p_value_bonferroni:.6f}")
                lines.append(f"- P-value (FDR): {test.ttest_p_value_fdr:.6f}")
                lines.append(f"- Cohen's d (effect size): {test.ttest_cohens_d:.6f}")
                lines.append("")

                lines.append("**Mann-Whitney U (Non-parametric)**:")
                lines.append(f"- Statistic: {test.mannwhitney_statistic:.6f}")
                lines.append(f"- P-value (uncorrected): {test.mannwhitney_p_value:.6f}")
                lines.append(f"- P-value (Bonferroni): {test.mannwhitney_p_value_bonferroni:.6f}")
                lines.append(f"- P-value (FDR): {test.mannwhitney_p_value_fdr:.6f}")
                lines.append(f"- Rank-biserial (effect size): {test.mannwhitney_rank_biserial:.6f}")
                lines.append("")

                lines.append("**Normality Tests (Shapiro-Wilk)**:")
                lines.append(f"- Group 1 p-value: {test.shapiro_p_value_group1:.6f}")
                lines.append(f"- Group 2 p-value: {test.shapiro_p_value_group2:.6f}")
                lines.append("")

                lines.append(
                    f"- **Sample Sizes**: Group 1: {test.n_group1}, Group 2: {test.n_group2}"
                )
                lines.append("")

    # Soil Saturation Contingency
    if summary.crosstab_results:
        lines.append("## Soil Saturation vs Event Occurrence")
        lines.append("")
        lines.append(
            "Cross-tabulation analysis between soil saturation levels (quartiles) and event occurrence."
        )
        lines.append("")

        # Group by season
        seasons = sorted(set(ct.season for ct in summary.crosstab_results))
        for season in seasons:
            season_crosstabs = [ct for ct in summary.crosstab_results if ct.season == season]
            if not season_crosstabs:
                continue

            lines.append(f"### Season: {season.upper()}")
            lines.append("")

            for crosstab in season_crosstabs:
                lines.append(f"#### {crosstab.row_variable} vs {crosstab.col_variable}")
                lines.append("")

                lines.append("**Contingency Table**:")
                lines.append("")

                # Build contingency table display
                if crosstab.contingency_table:
                    # Get unique row and column labels
                    row_labels = sorted(crosstab.contingency_table.keys())
                    col_labels = sorted(
                        set(col for row in crosstab.contingency_table.values() for col in row)
                    )

                    # Header
                    lines.append(
                        "| Soil Saturation | "
                        + " | ".join(f"Event={col}" for col in col_labels)
                        + " |"
                    )
                    lines.append("| --- | " + " | ".join(["---"] * len(col_labels)) + " |")

                    # Rows
                    for row_label in row_labels:
                        row_values = [
                            str(crosstab.contingency_table[row_label].get(col, 0))
                            for col in col_labels
                        ]
                        lines.append(f"| {row_label} | " + " | ".join(row_values) + " |")

                lines.append("")
                lines.append(f"- **Chi-square Statistic**: {crosstab.chi2_statistic:.6f}")
                lines.append(f"- **P-value (uncorrected)**: {crosstab.chi2_p_value:.6f}")
                lines.append(f"- **P-value (Bonferroni)**: {crosstab.chi2_p_value_bonferroni:.6f}")
                lines.append(f"- **P-value (FDR)**: {crosstab.chi2_p_value_fdr:.6f}")
                lines.append(f"- **Cramér's V (effect size)**: {crosstab.cramers_v:.6f}")
                lines.append(f"- **Sample Size**: {crosstab.n_samples}")
                lines.append("")

    # Onset Error Distribution
    if summary.onset_error_distributions:
        lines.append("## Onset Error Distribution")
        lines.append("")
        lines.append(
            "Distribution of onset errors (time between predicted and actual event start)."
        )
        lines.append("")
        lines.append("| Season | Min | Q10 | Q25 | Median | Mean | Q75 | Q90 | Max | N |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")

        for oed in sorted(summary.onset_error_distributions, key=lambda x: x.season):
            lines.append(
                f"| {oed.season} | {oed.min_hours:.2f} | {oed.p10_hours:.2f} | {oed.p25_hours:.2f} | "
                f"{oed.median_hours:.2f} | {oed.mean_hours:.2f} | {oed.p75_hours:.2f} | {oed.p90_hours:.2f} | "
                f"{oed.max_hours:.2f} | {oed.n_errors} |"
            )

        lines.append("")

    # Notes
    if summary.notes:
        lines.append("## Notes")
        lines.append("")
        for note in summary.notes:
            lines.append(f"- {note}")
        lines.append("")

    # Risk Assessment (Human-readable results)
    risk_lines = []
    risk_lines.append("## 🛡️ Ocena Ryzyka (Risk Assessment)")
    risk_lines.append("")
    risk_lines.append(
        "Na podstawie analizy statystycznej zidentyfikowano następujące czynniki wpływu:"
    )
    risk_lines.append("")

    # Logic to select rules based on summary
    significant_lags = [
        lc
        for lc in summary.lag_correlations
        if lc.pearson_p_value < 0.05 and abs(lc.pearson_r) > 0.1
    ]

    if any("rainfall" in lc.feature_name for lc in significant_lags):
        rule = EVENT_RULES.get(EventType.FLASH_FLOOD)
        if rule:
            risk_lines.append(f"### 🌧️ {rule.name}")
            risk_lines.append(f"**Komunikat**: {rule.response_message}")
            risk_lines.append("")

    if (
        any("soil_saturation" in lc.feature_name for lc in significant_lags)
        or summary.crosstab_results
    ):
        rule = EVENT_RULES.get(EventType.LONG_RAINFALL)
        if rule:
            risk_lines.append(f"### 💧 {rule.name}")
            risk_lines.append(f"**Komunikat**: {rule.response_message}")
            risk_lines.append("")

    if any("temperature" in lc.feature_name for lc in significant_lags):
        # Check if temperature correlation is positive (thaw) or negative (seasonal)
        temp_lags = [lc for lc in significant_lags if "temperature" in lc.feature_name]
        if any(lc.pearson_r > 0 for lc in temp_lags):
            rule = EVENT_RULES.get(EventType.THAW)
            if rule:
                risk_lines.append(f"### 🌡️ {rule.name}")
                risk_lines.append(f"**Komunikat**: {rule.response_message}")
                risk_lines.append("")
        else:
            rule = EVENT_RULES.get(EventType.SEASONAL_DEPENDENCY)
            if rule:
                risk_lines.append(f"### 📅 {rule.name}")
                risk_lines.append(f"**Komunikat**: {rule.response_message}")
                risk_lines.append("")

    lines.extend(risk_lines)

    lines.append("---")
    lines.append("*Raport wygenerowany automatycznie przez system MorzeAI.*")
    lines.append("")

    # Write report
    with Path.open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return risk_lines


def command(
    input_path: str,
    output_md: str | None = None,
    output_json: str | None = None,
    target_column: str = "water_level_m",
    event_column: str = "event_occurred",
    soil_saturation_column: str | None = None,
    threshold_percentile: float = 95.0,
    dataset_name: str = "unnamed",
) -> tuple[Any, Path, Path, list[str]]:
    """
    Run statistical analysis on predictions data.

    Args:
        input_path: Path to predictions CSV or evaluation.json
        output_md: Output markdown report path
        output_json: Output JSON summary path
        target_column: Target variable column name
        event_column: Event binary column name
        soil_saturation_column: Soil saturation column name
        threshold_percentile: Percentile cutoff for high water
        dataset_name: Name for reporting

    Returns:
        Tuple of (StatisticalSummary, markdown_path, json_path, risk_assessment_lines)
    """
    input_file = Path(input_path)

    # Determine output paths
    if output_md is None:
        output_md = str(input_file.parent / "statistical_analysis.md")
    if output_json is None:
        output_json = str(input_file.parent / "statistical_analysis.json")

    output_md_path = Path(output_md)
    output_json_path = Path(output_json)

    # Load data
    if input_file.suffix == ".json":
        # Load evaluation.json and extract predictions
        payload = read_json(input_file)
        if "predictions" in payload:
            df = pd.DataFrame(payload["predictions"])
        else:
            raise ValueError("JSON file must contain 'predictions' key with data")
    elif input_file.suffix == ".csv":
        df = read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")

    # Initialize analyzer
    analyzer = StatisticalAnalyzer(df, dataset_name=dataset_name)

    # Detect lag columns
    lag_columns = [col for col in df.columns if "_lag_" in col and col.endswith("h")]

    # Generate statistical summary
    stat_summary = analyzer.generate_statistical_summary(
        target_column=target_column,
        event_column=event_column,
        soil_saturation_column=soil_saturation_column
        if soil_saturation_column
        else "soil_saturation_index",
        features_to_test=[
            col
            for col in df.columns
            if col.startswith(("rainfall", "temperature", "pressure", "soil_saturation"))
            and "_lag_" not in col
        ],
        lag_columns=lag_columns if lag_columns else None,
        threshold_percentile=threshold_percentile,
    )

    # Save markdown report
    risk_assessment = generate_markdown_report(stat_summary, output_md_path)

    # Save JSON summary
    ensure_parent(output_json_path)
    import json

    with Path.open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(stat_summary.to_dict(), f, indent=2, default=str)

    return stat_summary, output_md_path, output_json_path, risk_assessment


def main() -> None:
    """Parse command-line arguments and run statistical analysis."""
    args = build_parser().parse_args()

    summary, md_path, _json_path, risk_assessment = command(
        input_path=args.input,
        output_md=args.output_md,
        output_json=args.output_json,
        target_column=args.target_column,
        event_column=args.event_column,
        soil_saturation_column=args.soil_saturation_column,
        threshold_percentile=args.threshold_percentile,
        dataset_name=args.dataset_name,
    )

    print(f"\n✓ Analysis complete: {md_path}")

    if risk_assessment:
        print("\n┌──────────────────────────────────────────┐")
        print("│        RISK ASSESSMENT (OCENA RYZYKA)    │")
        print("└──────────────────────────────────────────┘")
        for line in risk_assessment:
            clean_line = line.replace("### ", "• ").replace("**", "")
            if clean_line.strip():
                print(f"  {clean_line}")
        print("────────────────────────────────────────────")

    print("\n  Summary statistics:")
    print(f"    • {len(summary.lag_correlations)} lag correlations")
    print(f"    • {len(summary.hypothesis_tests)} hypothesis tests")
    print(f"    • {len(summary.crosstab_results)} contingency analyses")
    print(f"    • {len(summary.onset_error_distributions)} onset error distributions")

    if summary.warnings:
        print(f"\n  ⚠️ Warnings ({len(summary.warnings)}):")
        for warning in summary.warnings:
            print(f"    - {warning}")


if __name__ == "__main__":
    main()
