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
ERA5_PROCESSED = ROOT / "data" / "processed" / "era5_hourly_full.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "water_level_training_with_wind.csv"


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

    # Ensure ERA5-derived hourly file exists; try to create it if missing
    if not ERA5_PROCESSED.exists():
        try:
            from src.data.era5_processor import process_all_years

            print("ERA5 processed CSV not found; attempting to create from raw netCDF files...")
            process_all_years()
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"Warning: could not create ERA5 processed CSV: {exc}")

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

    # Merge ERA5-derived features when available
    if ERA5_PROCESSED.exists():
        try:
            era5 = pd.read_csv(ERA5_PROCESSED)
            era5["timestamp"] = pd.to_datetime(era5["timestamp"], errors="coerce")
            training = training.merge(era5, on="timestamp", how="left")
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"Warning: could not merge ERA5 data: {exc}")

    # Consolidate duplicated pressure columns from weather + ERA5 sources.
    if "pressure_hpa_x" in training.columns or "pressure_hpa_y" in training.columns:
        pressure_candidates = [
            column for column in ["pressure_hpa_x", "pressure_hpa_y"] if column in training.columns
        ]
        training["pressure_hpa"] = training[pressure_candidates].bfill(axis=1).iloc[:, 0]
        training = training.drop(columns=pressure_candidates)

    training = training.dropna(subset=["timestamp", "water_level_m"])

    numeric_cols = [
        "rainfall_mm",
        "temperature_c",
        "humidity_percentage",
        "pressure_hpa",
        "wind_speed_ms",
        "wind_u",
        "wind_v",
        "sea_surface_temperature_c",
    ]

    available_numeric = [c for c in numeric_cols if c in training.columns]
    for column in available_numeric:
        training[column] = pd.to_numeric(training[column], errors="coerce")

    if available_numeric:
        training[available_numeric] = (
            training[available_numeric].interpolate(limit_direction="both").fillna(0.0)
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
