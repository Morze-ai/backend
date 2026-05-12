from __future__ import annotations

import pandas as pd

from src.data.meteorological_aggregation import (
    aggregate_meteorological_hourly,
    create_daily_meteorological_aggregations,
    validate_meteorological_alignment,
)


def test_aggregate_meteorological_hourly_uses_sum_for_rainfall() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:10:00",
                    "2024-01-01 00:40:00",
                    "2024-01-01 01:15:00",
                ]
            ),
            "rainfall_mm": [1.0, 2.0, 3.0],
            "temperature_c": [10.0, 14.0, 8.0],
        }
    )

    hourly = aggregate_meteorological_hourly(frame)

    first_hour = hourly.loc[hourly["timestamp"] == pd.Timestamp("2024-01-01 00:00:00")]
    assert not first_hour.empty
    assert float(first_hour["rainfall_mm"].iloc[0]) == 3.0
    assert float(first_hour["temperature_c"].iloc[0]) == 12.0


def test_create_daily_meteorological_aggregations_creates_expected_columns() -> None:
    hourly = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="h"),
            "rainfall_mm": [1.0] * 30,
            "temperature_c": [10.0] * 30,
        }
    )

    daily = create_daily_meteorological_aggregations(hourly)

    assert "rainfall_mm_sum" in daily.columns
    assert "temperature_c_mean" in daily.columns
    assert "temperature_c_max" in daily.columns
    assert "temperature_c_min" in daily.columns
    assert "date_str" in daily.columns


def test_validate_meteorological_alignment_reports_gaps() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 03:00:00"]
            ),
            "rainfall_mm": [0.0, 1.0, 2.0],
        }
    )

    stats = validate_meteorological_alignment(frame)
    assert stats["missing_timestamps"] == 1
    assert stats["total_rows"] == 3
