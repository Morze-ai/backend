"""Tests for the rule-based event detector system."""

from __future__ import annotations

import pandas as pd

from src.events.detectors import (
    detect_flash_flood,
    detect_long_rainfall,
    detect_seasonal_dependencies,
    detect_thaw,
)


def _hourly_frame(
    start: str, rows: int, rainfall: list[float], temperature: list[float] | None = None
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=rows, freq="h"),
            "rainfall_mm": rainfall,
        }
    )
    if temperature is not None:
        frame["temperature_c"] = temperature
    return frame


def test_detect_long_rainfall() -> None:
    """Long rainfall should be detected when 72h and 7d sums exceed thresholds."""
    rainfall = [0.2] * 120
    rainfall[-100:] = [1.0] * 100
    rainfall[-1] = 5.5
    frame = _hourly_frame("2026-01-01", 120, rainfall)
    detection = detect_long_rainfall(frame)
    assert detection.detected is True
    assert detection.event_type.value == "long_rainfall"
    assert "nasycona" in detection.message
    assert detection.metadata["rainfall_72h_mm"] >= 40.0
    assert detection.metadata["rainfall_7d_mm"] >= 90.0


def test_detect_flash_flood() -> None:
    """Flash flood should be detected for a short, intense rainfall burst."""
    rainfall = [0.0] * 30
    rainfall[-12:] = [0.5] * 12
    rainfall[-1] = 12.0
    frame = _hourly_frame("2026-01-01", 30, rainfall)
    detection = detect_flash_flood(frame)
    assert detection.detected is True
    assert detection.event_type.value == "flash_flood"
    assert "gwa" in detection.message.lower()
    assert detection.metadata["rainfall_mm"] >= 12.0
    assert detection.metadata["positive_count"] >= 12


def test_detect_thaw() -> None:
    """Thaw should be detected in winter-spring when the temperature rises above zero quickly."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-02-20", periods=10, freq="D"),
            "temperature_c": [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0],
        }
    )
    detection = detect_thaw(frame)
    assert detection.detected is True
    assert detection.event_type.value == "thaw"
    assert "roztop" in detection.message.lower()
    assert detection.metadata["temperature_increase_c"] >= 5.0


def test_detect_seasonal_dependencies() -> None:
    """Seasonal dependency should label the active seasonal driver."""
    rainfall = [0.2] * 60
    rainfall[-20:] = [6.0] * 20
    temperatures = [21.0] * 60
    frame = _hourly_frame("2026-07-01", 60, rainfall, temperatures)
    frame["water_level_m"] = [1.0 + i * 0.01 for i in range(60)]
    detection = detect_seasonal_dependencies(frame)
    assert detection.detected is True
    assert detection.event_type.value == "seasonal_dependency"
    assert "sezon" in detection.message.lower()
    assert detection.metadata["dominant_factor"] in {"opadowy", "roztopowy", "nasycenie zlewni"}
