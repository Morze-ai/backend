"""Aggregation helpers for meteorological datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd


def aggregate_meteorological_hourly(
    weather_df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Aggregate meteorological observations to hourly granularity.

    Rainfall is summed per hour, while the remaining numeric variables
    are averaged.
    """

    if timestamp_column not in weather_df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in weather data")

    result = weather_df.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], errors="coerce")
    result = result.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
    result = result.set_index(timestamp_column)

    numeric_columns = [
        column for column in result.columns if pd.api.types.is_numeric_dtype(result[column])
    ]

    if not numeric_columns:
        raise ValueError("No numeric meteorological columns available for aggregation")

    hourly = pd.DataFrame(index=result.resample("h").size().index)
    for column in numeric_columns:
        if column.startswith("rain") or column == "rainfall_mm":
            hourly[column] = result[column].resample("h").sum()
        else:
            hourly[column] = result[column].resample("h").mean()

    hourly = hourly.reset_index().rename(columns={"index": timestamp_column})
    return hourly


def create_daily_meteorological_aggregations(
    hourly_df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Create daily weather aggregates for reporting/verification."""

    if timestamp_column not in hourly_df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in hourly weather data")

    result = hourly_df.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], errors="coerce")
    result = result.dropna(subset=[timestamp_column]).set_index(timestamp_column).sort_index()

    numeric_columns = [
        column for column in result.columns if pd.api.types.is_numeric_dtype(result[column])
    ]
    if not numeric_columns:
        raise ValueError("No numeric columns available for daily meteorological aggregation")

    aggregated = pd.DataFrame(index=result.resample("D").size().index)
    for column in numeric_columns:
        if column.startswith("rain") or column == "rainfall_mm":
            aggregated[f"{column}_sum"] = result[column].resample("D").sum()
        else:
            aggregated[f"{column}_mean"] = result[column].resample("D").mean()
            aggregated[f"{column}_max"] = result[column].resample("D").max()
            aggregated[f"{column}_min"] = result[column].resample("D").min()

    aggregated = aggregated.reset_index().rename(columns={timestamp_column: "date"})
    aggregated["date_str"] = aggregated["date"].dt.strftime("%Y-%m-%d")
    aggregated["year"] = aggregated["date"].dt.year
    aggregated["month"] = aggregated["date"].dt.month
    aggregated["day_of_year"] = aggregated["date"].dt.dayofyear

    return aggregated


def validate_meteorological_alignment(
    hourly_df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> dict[str, Any]:
    """Return basic quality stats for the hourly weather frame."""

    if timestamp_column not in hourly_df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in hourly weather data")

    result = hourly_df.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], errors="coerce")
    result = result.dropna(subset=[timestamp_column]).sort_values(timestamp_column)

    start_time = result[timestamp_column].min()
    end_time = result[timestamp_column].max()
    expected = pd.date_range(start=start_time, end=end_time, freq="h")

    return {
        "start_time": start_time,
        "end_time": end_time,
        "total_rows": len(result),
        "inferred_frequency": pd.infer_freq(result[timestamp_column]) if len(result) > 1 else None,
        "missing_timestamps": int(max(0, len(expected) - len(result))),
    }
