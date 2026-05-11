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

import pandas as pd

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
        Strategy name ("valid-zero" or "invalid-zero")
    """
    name = dataset_path.stem.lower()

    if "vistula" in name or "dead" in name or "strzyza" in name:
        return "valid-zero"
    elif "port" in name or "northern" in name:
        return "invalid-zero"
    else:
        return "invalid-zero"  # Default to invalid-zero strategy
    # Invalid-zero strategy replaces both "-" and "0" with NaN.
    # This shouldn't be a problem as other datasets shouldn't have "0" as a valid value.
    # Adjust if needed.


def clean_river_dataset(input_path: Path, output_path: Path) -> dict:
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

    # Count missing before (convert to numeric first to detect literal zeros)
    missing_dashes = (df["water_level_m"] == "-").sum()
    numeric_col = pd.to_numeric(df["water_level_m"], errors="coerce")
    missing_zeros = (numeric_col == 0).sum() if strategy.treat_zero_as_missing else 0
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


def clean_weather_dataset(input_path: Path, output_path: Path) -> dict:
    """
    Clean weather dataset, dropping problematic columns.

    For hail mountain data: drops rainfall_mm column (mostly 0s, unreliable)
    and converts "-" to NaN in remaining columns.

    Returns:
        Statistics dictionary with processing info
    """
    print(f"\n{'=' * 70}")
    print(f"Processing: {input_path.name}")
    print(f"{'=' * 70}")

    df = pd.read_csv(input_path)

    print(f"Dataset: {input_path.name}")
    print("Type: Weather Data")
    print("  -> Dropping rainfall_mm column (unreliable data quality)")
    print(f"Total rows: {len(df)}")
    print(f"Original columns: {list(df.columns)}")

    # Drop rainfall column
    if "rainfall_mm" in df.columns:
        df = df.drop(columns=["rainfall_mm"])

    # Convert "-" to NaN and then forward-fill any gaps
    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col].replace("-", pd.NA), errors="coerce")

    print(f"Cleaned columns: {list(df.columns)}")
    print(f"Rows after cleaning: {len(df)}")

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return {
        "dataset": "weather",
        "input_rows": len(df),
        "output_rows": len(df),
        "columns_dropped": ["rainfall_mm"],
    }


if __name__ == "__main__":
    results = []

    # Clean Dead Vistula River dataset
    results.append(
        clean_river_dataset(
            RAW_DATA_DIR / "dead-vistula-river-water-level-2021-2025.csv",
            PROCESSED_DATA_DIR / "dead-vistula-cleaned.csv",
        )
    )

    # Clean Northern Port dataset
    results.append(
        clean_river_dataset(
            RAW_DATA_DIR / "northern-port-water-level-2021-2025.csv",
            PROCESSED_DATA_DIR / "northern-port-cleaned.csv",
        )
    )

    # Clean Strzyza River dataset
    results.append(
        clean_river_dataset(
            RAW_DATA_DIR / "strzyza-river-water-level-2021-2025.csv",
            PROCESSED_DATA_DIR / "strzyza-river-cleaned.csv",
        )
    )

    # Clean Hail Mountain weather dataset
    results.append(
        clean_weather_dataset(
            RAW_DATA_DIR / "hail-mountain-weather-data-2021-2025.csv",
            PROCESSED_DATA_DIR / "hail-mountain-weather-cleaned.csv",
        )
    )

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for result in results:
        if "missing_before" in result:
            print(f"\n{result['dataset'].upper()}:")
            print(f"  Total rows: {result['input_rows']}")
            print(f"  Missing before: {result['missing_before']}")
            print(f"  Missing after: {result['missing_after']}")
            print(f"  Imputed: {result['pct_imputed']}%")
        else:
            print(f"\n{result['dataset'].upper()}:")
            print(f"  Total rows: {result['input_rows']}")
            print(f"  Columns dropped: {result['columns_dropped']}")
