#!/usr/bin/env python
"""
Script to clean missing values in water level datasets.

Handles missing values (marked as "-" and sometimes "0" for port) with strategies:
- Small gaps (1-6 measurements): nearest neighbor mean
- Large gaps (>6 measurements): seasonal averaging (same dates from other years)

Uses dataset metadata to determine appropriate imputation strategy per dataset.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_dataset
from src.data.preprocessing import IMPUTATION_STRATEGIES, handle_missing_values

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def infer_strategy_from_dataset(dataset_path: Path) -> str:
    """
    Infer imputation strategy based on dataset filename or metadata.

    Args:
        dataset_path: Path to the CSV file

    Returns:
        Strategy name ("vistula" or "port")
    """
    name = dataset_path.stem.lower()

    if "vistula" in name or "dead" in name:
        return "vistula"
    elif "port" in name or "northern" in name:
        return "port"
    else:
        return "port"  # Default to port strategy
    # Port strategy replaces both "-" and "0" with NaN.
    # This shouldn't be a problem as other datasets shouldn't have "0" as a valid value.
    # Adjust if needed.


def clean_dataset(input_path: Path, output_path: Path) -> dict:
    """
    Clean missing values in a dataset and save results.

    Uses metadata and filename to determine appropriate strategy.

    Returns:
        Statistics dictionary with before/after metrics
    """
    print(f"\n{'=' * 70}")
    print(f"Processing: {input_path.name}")
    print(f"{'=' * 70}")

    # Load dataset with metadata
    artifact = load_dataset(input_path)
    df = artifact.frame

    # Infer strategy
    strategy_name = infer_strategy_from_dataset(input_path)
    strategy = IMPUTATION_STRATEGIES[strategy_name]

    print(f"Dataset: {input_path.name}")
    print(f"Strategy: {strategy_name}")
    if strategy.treat_zero_as_missing:
        print("  -> Treating 0 as missing values")
    else:
        print("  -> Preserving 0 values")

    # Count missing before
    missing_dashes = (df["water_level_m"] == "-").sum()
    missing_zeros = (df["water_level_m"] == 0).sum() if strategy.treat_zero_as_missing else 0
    missing_before = missing_dashes + missing_zeros

    print(f"\nTotal rows: {len(df)}")
    print(f"Missing values (marked as '-'): {missing_dashes}")
    if strategy.treat_zero_as_missing:
        print(f"Missing values (marked as '0'): {missing_zeros}")

    # Clean missing values
    df_cleaned = handle_missing_values(df, strategy=strategy)

    missing_after = df_cleaned["water_level_m"].isna().sum()
    imputed_count = missing_before - missing_after
    pct_imputed = round(100 * imputed_count / max(missing_before, 1), 2)

    print(f"\nRemaining NaN after cleaning: {missing_after}")
    print(f"Successfully imputed: {imputed_count} ({pct_imputed}%)")

    # Save cleaned dataset
    df_cleaned.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return {
        "dataset": strategy_name,
        "input_rows": len(df),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "pct_imputed": pct_imputed,
    }


if __name__ == "__main__":
    results = []

    # Clean Dead Vistula River dataset
    results.append(
        clean_dataset(
            RAW_DATA_DIR / "dead-vistula-river-water-level-2021-2025.csv",
            PROCESSED_DATA_DIR / "dead-vistula-cleaned.csv",
        )
    )

    # Clean Northern Port dataset
    results.append(
        clean_dataset(
            RAW_DATA_DIR / "northern-port-water-level-2021-2025.csv",
            PROCESSED_DATA_DIR / "northern-port-cleaned.csv",
        )
    )

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for result in results:
        print(f"\n{result['dataset'].upper()}:")
        print(f"  Total rows: {result['input_rows']}")
        print(f"  Missing before: {result['missing_before']}")
        print(f"  Missing after: {result['missing_after']}")
        print(f"  Imputed: {result['pct_imputed']}%")
