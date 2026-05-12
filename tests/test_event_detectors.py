import numpy as np
import pandas as pd

from src.events.detectors.rainfall import detect_flash_flood, detect_long_rainfall
from src.events.detectors.seasonal import detect_seasonal_dependencies
from src.events.detectors.thaw import detect_thaw


def make_hourly_df(start: str, hours: int, vals: dict) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=hours, freq="h")
    df = pd.DataFrame(index=idx)
    for k, v in vals.items():
        df[k] = v
    df = df.reset_index().rename(columns={"index": "timestamp"})
    return df


def test_detect_flash_flood_triggers_on_high_6h_sum():
    # create 48h of low rainfall, then a big spike over 6h
    hours = 48
    rain = np.zeros(hours)
    # spike in last 6 hours
    rain[-6:] = 10.0
    df = make_hourly_df("2024-01-01", hours, {"rainfall_mm": rain})

    res = detect_flash_flood(df)
    assert res.detected is True
    assert res.confidence is not None and res.confidence > 0.0


def test_detect_long_rainfall_sustained_event():
    # 8 days of light but continuous rainfall -> will exceed 72h and 7d thresholds
    hours = 8 * 24
    rain = np.ones(hours) * 1.0
    df = make_hourly_df(
        "2023-10-01", hours, {"rainfall_mm": rain, "soil_saturation_index": np.full(hours, 0.8)}
    )

    res = detect_long_rainfall(df)
    assert res.detected is True
    assert res.confidence is not None and res.confidence >= 0.0
    assert "soil_saturation_index" in res.metadata


def test_detect_thaw_simple_case():
    # 10 days: first 7 days below zero, last day above zero
    hours = 10 * 24
    temps = np.concatenate([np.full(7 * 24, -2.0), np.full(3 * 24, 3.0)])
    df = make_hourly_df(
        "2024-03-01", hours, {"temperature_c": temps, "wind_speed_ms": np.zeros(hours)}
    )

    res = detect_thaw(df)
    assert res.detected is True
    assert res.confidence is not None and res.confidence > 0.0


def test_detect_seasonal_dependencies_identifies_dominant_driver():
    # Create a seasonal slice where temperature strongly correlates with water level
    hours = 30 * 24
    rng = pd.date_range(start="2024-06-01", periods=hours, freq="h")
    temp = np.linspace(10, 20, hours)
    water = temp * 0.5 + np.random.normal(0, 0.1, hours)
    df = pd.DataFrame(
        {
            "timestamp": rng,
            "temperature_c": temp,
            "water_level_m": water,
            "rainfall_mm": np.zeros(hours),
        }
    )

    res = detect_seasonal_dependencies(df)
    assert res.detected is True
    assert res.metadata.get("dominant_factor") in {"temperature_c", "rainfall_mm", "wind_speed_ms"}
