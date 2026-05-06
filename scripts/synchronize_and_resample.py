#!/usr/bin/env python
"""
Script to synchronize and resample water level datasets.

Synchronizes Vistula and Port water level data to common hourly and daily frequencies.
Creates two outputs:
1. Hourly merged view (for detailed analysis)
2. Daily aggregated statistics (mean, max, min for seasonal patterns)
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synchronization import (
    create_daily_aggregations,
    merge_datasets,
    validate_alignment,
)

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def synchronize_and_resample() -> None:
    """Main orchestration function: synchronizes datasets and creates daily aggregations."""
    vistula_path = RAW_DATA_DIR / "dead-vistula-cleaned.csv"
    port_path = RAW_DATA_DIR / "northern-port-cleaned.csv"

    print("=" * 70)
    print("SYNCHRONIZING WATER LEVEL DATASETS")
    print("=" * 70)

    # Validate input files exist
    if not vistula_path.exists():
        print(f"ERROR: Vistula cleaned CSV not found at {vistula_path}")
        sys.exit(1)

    if not port_path.exists():
        print(f"ERROR: Port cleaned CSV not found at {port_path}")
        sys.exit(1)

    print("\nLoading datasets:")
    print(f"  Vistula: {vistula_path.name}")
    print(f"  Port: {port_path.name}")

    # Merge datasets
    print("\n" + "-" * 70)
    print("PHASE 1: MERGING DATASETS")
    print("-" * 70)
    merged = merge_datasets(vistula_path=str(vistula_path), port_path=str(port_path))

    # Validate alignment
    alignment_stats = validate_alignment(merged)

    print("\nAlignment validation:")
    print(f"  Time range: {alignment_stats['start_time']} to {alignment_stats['end_time']}")
    print(f"  Total rows: {alignment_stats['total_rows']:,}")
    print(f"  Inferred frequency: {alignment_stats['inferred_frequency']}")
    print(f"  Missing timestamps (gaps): {alignment_stats['missing_timestamps']}")
    print("\nData integrity:")
    print(f"  NaN in Vistula: {alignment_stats['nan_vistula']}")
    print(f"  NaN in Port: {alignment_stats['nan_port']}")
    print("\nData ranges:")
    print(
        f"  Vistula: [{alignment_stats['vistula_range'][0]:.2f}, {alignment_stats['vistula_range'][1]:.2f}] m"
    )
    print(
        f"  Port: [{alignment_stats['port_range'][0]:.2f}, {alignment_stats['port_range'][1]:.2f}] m"
    )

    # Save hourly merged dataset
    hourly_output_path = OUTPUT_DIR / "water_level_synchronized_hourly.csv"
    merged.to_csv(hourly_output_path, index=False)
    print(f"\n✓ Hourly synchronized dataset saved: {hourly_output_path.name}")

    # Create daily aggregations
    print("\n" + "-" * 70)
    print("PHASE 2: RESAMPLING TO DAILY AGGREGATIONS")
    print("-" * 70)
    daily = create_daily_aggregations(merged)

    print("\nDaily aggregations created:")
    print(f"  Total days: {len(daily):,}")
    print(f"  Date range: {daily['date_str'].min()} to {daily['date_str'].max()}")
    print(f"  Columns: {len(daily.columns)}")

    # Verify daily stats make sense (min <= mean <= max)
    vistula_valid = (
        (daily["vistula_min_m"] <= daily["vistula_mean_m"])
        & (daily["vistula_mean_m"] <= daily["vistula_max_m"])
    ).sum()
    port_valid = (
        (daily["port_min_m"] <= daily["port_mean_m"])
        & (daily["port_mean_m"] <= daily["port_max_m"])
    ).sum()

    print("\nDaily statistics validation:")
    print(f"  Vistula valid (min ≤ mean ≤ max): {vistula_valid}/{len(daily)}")
    print(f"  Port valid (min ≤ mean ≤ max): {port_valid}/{len(daily)}")

    # Save daily aggregated dataset
    daily_output_path = OUTPUT_DIR / "water_level_daily.csv"
    daily.to_csv(daily_output_path, index=False)
    print(f"\n✓ Daily aggregated dataset saved: {daily_output_path.name}")

    # Print sample data
    print("\n" + "-" * 70)
    print("SAMPLE DATA (First 5 days)")
    print("-" * 70)
    print(
        daily[["date_str", "vistula_mean_m", "vistula_max_m", "port_mean_m", "port_max_m"]].head()
    )

    print("\n" + "=" * 70)
    print("SYNCHRONIZATION COMPLETE")
    print("=" * 70)
    print("\nOutputs:")
    print(f"  Hourly: {hourly_output_path}")
    print(f"  Daily: {daily_output_path}")


if __name__ == "__main__":
    synchronize_and_resample()
