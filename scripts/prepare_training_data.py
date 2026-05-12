#!/usr/bin/env python
"""Prepare a labeled training dataset from the synchronized hourly water-level data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import read_csv_safe, save_csv_with_metadata

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "data" / "processed" / "water_level_synchronized_hourly.csv"
WEATHER_PATH = ROOT / "data" / "raw" / "hail-mountain-weather-data-2021-2025.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "water_level_training.csv"


def main() -> None:
    """Create a simple binary classification dataset from the synchronized hourly series."""

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing synchronized source data at {SOURCE_PATH}. Run `make synchronize-data` first."
        )
    if not WEATHER_PATH.exists():
        raise FileNotFoundError(
            f"Missing weather source data at {WEATHER_PATH}. The training dataset needs it for feature engineering."
        )

    water_artifact = read_csv_safe(SOURCE_PATH)
    weather_artifact = read_csv_safe(WEATHER_PATH)
    frame = water_artifact.frame.copy()
    weather = weather_artifact.frame.copy()

    required_columns = {"timestamp", "vistula_water_level_m"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Source data is missing required columns: {missing}")

    weather_required = {
        "timestamp",
        "rainfall_mm",
        "temperature_c",
        "humidity_percentage",
        "pressure_hpa",
    }
    weather_missing = sorted(weather_required - set(weather.columns))
    if weather_missing:
        raise ValueError(f"Weather source data is missing required columns: {weather_missing}")

    training = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], errors="coerce"),
            "water_level_m": pd.to_numeric(frame["vistula_water_level_m"], errors="coerce"),
        }
    )
    training["timestamp"] = pd.to_datetime(training["timestamp"], errors="coerce")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce")
    training = training.merge(weather, on="timestamp", how="inner")
    training = training.dropna(subset=["timestamp", "water_level_m"])

    for column in ["rainfall_mm", "temperature_c", "humidity_percentage", "pressure_hpa"]:
        training[column] = pd.to_numeric(training[column], errors="coerce")

    training[["rainfall_mm", "temperature_c", "humidity_percentage", "pressure_hpa"]] = (
        training[["rainfall_mm", "temperature_c", "humidity_percentage", "pressure_hpa"]]
        .interpolate(limit_direction="both")
        .fillna(0.0)
    )

    threshold = float(training["water_level_m"].median())
    training["water_level_class"] = training["water_level_m"].apply(
        lambda value: "high" if float(value) >= threshold else "low"
    )

    save_csv_with_metadata(
        training,
        OUTPUT_PATH,
        source=str(SOURCE_PATH),
        description="Binary water-level classification dataset derived from the synchronized hourly series.",
        extras={"threshold_median": threshold, "target_column": "water_level_class"},
    )

    print(f"✓ Training dataset saved: {OUTPUT_PATH}")
    print(f"✓ Median threshold used for labels: {threshold:.4f} m")


if __name__ == "__main__":
    main()
