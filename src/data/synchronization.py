"""Synchronizes and resamples water level datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd


def merge_datasets(
    vistula_path: str | None = None,
    port_path: str | None = None,
    strzyza_path: str | None = None,
    vistula_df: pd.DataFrame | None = None,
    port_df: pd.DataFrame | None = None,
    strzyza_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merges Vistula, Port, and Strzyza water level datasets on timestamp.

    Loads datasets from paths if provided, otherwise uses DataFrame objects directly.
    Renames water_level_m columns to distinguish between sources and performs inner join
    to keep only overlapping timestamps.

    Args:
        vistula_path: Path to Vistula cleaned CSV file.
        port_path: Path to Port cleaned CSV file.
        strzyza_path: Path to Strzyza cleaned CSV file.
        vistula_df: Pre-loaded Vistula DataFrame (used if vistula_path is None).
        port_df: Pre-loaded Port DataFrame (used if port_path is None).
        strzyza_df: Pre-loaded Strzyza DataFrame (used if strzyza_path is None).

    Returns:
        pd.DataFrame: Merged dataset with columns [timestamp, vistula_water_level_m, port_water_level_m].

    Raises:
        ValueError: If neither paths nor DataFrames are provided, or if merge results in empty DataFrame.
    """
    # Load datasets if paths provided
    if vistula_path is not None:
        vistula_df = pd.read_csv(vistula_path)
    if port_path is not None:
        port_df = pd.read_csv(port_path)
    if strzyza_path is not None:
        strzyza_df = pd.read_csv(strzyza_path)

    if vistula_df is None or port_df is None or strzyza_df is None:
        raise ValueError("Must provide either paths or DataFrame objects for all datasets")

    # Ensure timestamp is datetime
    vistula_df = vistula_df.copy()
    port_df = port_df.copy()
    strzyza_df = strzyza_df.copy()
    vistula_df["timestamp"] = pd.to_datetime(vistula_df["timestamp"])
    port_df["timestamp"] = pd.to_datetime(port_df["timestamp"])
    strzyza_df["timestamp"] = pd.to_datetime(strzyza_df["timestamp"])

    # Rename water level columns to distinguish sources
    vistula_df = vistula_df.rename(columns={"water_level_m": "vistula_water_level_m"})
    port_df = port_df.rename(columns={"water_level_m": "port_water_level_m"})
    strzyza_df = strzyza_df.rename(columns={"water_level_m": "strzyza_water_level_m"})

    # Ensure numeric types for water level
    vistula_df["vistula_water_level_m"] = pd.to_numeric(
        vistula_df["vistula_water_level_m"], errors="coerce"
    )
    port_df["port_water_level_m"] = pd.to_numeric(port_df["port_water_level_m"], errors="coerce")
    strzyza_df["strzyza_water_level_m"] = pd.to_numeric(
        strzyza_df["strzyza_water_level_m"], errors="coerce"
    )

    # Merge on timestamp with inner join (keep only overlapping times)
    merged = pd.merge(
        vistula_df[["timestamp", "vistula_water_level_m"]],
        port_df[["timestamp", "port_water_level_m"]],
        on="timestamp",
        how="inner",
    )
    merged = pd.merge(
        merged,
        strzyza_df[["timestamp", "strzyza_water_level_m"]],
        on="timestamp",
        how="inner",
    )

    if merged.empty:
        raise ValueError("Merged dataset is empty; no overlapping timestamps found")

    # Sort by timestamp for clarity
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    return merged


def validate_alignment(merged_df: pd.DataFrame) -> dict[str, Any]:
    """Validates the alignment and frequency of merged dataset.

    Args:
        merged_df: Merged dataset with timestamp column.

    Returns:
        dict: Alignment statistics including time range, frequency, gaps, and NaN counts.
    """
    merged_df = merged_df.copy()
    merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"])

    # Check time range
    start_time = merged_df["timestamp"].min()
    end_time = merged_df["timestamp"].max()
    total_rows = len(merged_df)

    # Check frequency
    inferred_freq = pd.infer_freq(merged_df["timestamp"]) if len(merged_df) > 1 else None

    # Check for gaps (missing timestamps)
    expected_hourly = pd.date_range(start=start_time, end=end_time, freq="h")
    gaps = len(expected_hourly) - len(merged_df)

    # Check for NaN
    nan_vistula = merged_df["vistula_water_level_m"].isna().sum()
    nan_port = merged_df["port_water_level_m"].isna().sum()
    nan_strzyza = merged_df["strzyza_water_level_m"].isna().sum()
    # Data range statistics
    stats = {
        "start_time": start_time,
        "end_time": end_time,
        "total_rows": total_rows,
        "inferred_frequency": inferred_freq,
        "missing_timestamps": gaps,
        "nan_vistula": nan_vistula,
        "nan_port": nan_port,
        "nan_strzyza": nan_strzyza,
        "vistula_range": (
            merged_df["vistula_water_level_m"].min(),
            merged_df["vistula_water_level_m"].max(),
        ),
        "port_range": (
            merged_df["port_water_level_m"].min(),
            merged_df["port_water_level_m"].max(),
        ),
        "strzyza_range": (
            merged_df["strzyza_water_level_m"].min(),
            merged_df["strzyza_water_level_m"].max(),
        ),
    }

    return stats


def create_daily_aggregations(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Creates daily aggregations (mean, max, min) from hourly merged data.

    Args:
        merged_df: Hourly merged dataset with timestamp and water level columns.

    Returns:
        pd.DataFrame: Daily aggregated dataset with columns:
            - date: YYYY-MM-DD
            - year, month, day_of_year: For seasonal analysis
            - vistula_mean_m, vistula_max_m, vistula_min_m
            - port_mean_m, port_max_m, port_min_m
            - strzyza_mean_m, strzyza_max_m, strzyza_min_m
    """
    merged_df = merged_df.copy()
    merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"])
    merged_df = merged_df.set_index("timestamp")

    # Create daily aggregations
    daily = pd.DataFrame()

    # Mean values
    daily["vistula_mean_m"] = merged_df["vistula_water_level_m"].resample("D").mean().round(2)
    daily["port_mean_m"] = merged_df["port_water_level_m"].resample("D").mean().round(2)
    daily["strzyza_mean_m"] = merged_df["strzyza_water_level_m"].resample("D").mean().round(2)

    # Max values
    daily["vistula_max_m"] = merged_df["vistula_water_level_m"].resample("D").max()
    daily["port_max_m"] = merged_df["port_water_level_m"].resample("D").max()
    daily["strzyza_max_m"] = merged_df["strzyza_water_level_m"].resample("D").max()

    # Min values
    daily["vistula_min_m"] = merged_df["vistula_water_level_m"].resample("D").min()
    daily["port_min_m"] = merged_df["port_water_level_m"].resample("D").min()
    daily["strzyza_min_m"] = merged_df["strzyza_water_level_m"].resample("D").min()

    # Reset index to make date a column
    daily = daily.reset_index()
    daily = daily.rename(columns={"timestamp": "date"})

    # Add metadata columns for seasonal analysis
    daily["date_str"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.dayofyear

    # Reorder columns for clarity
    daily = daily[
        [
            "date",
            "date_str",
            "year",
            "month",
            "day_of_year",
            "vistula_mean_m",
            "vistula_max_m",
            "vistula_min_m",
            "port_mean_m",
            "port_max_m",
            "port_min_m",
            "strzyza_mean_m",
            "strzyza_max_m",
            "strzyza_min_m",
        ]
    ]

    return daily
