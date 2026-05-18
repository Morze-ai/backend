"""Detects season-driven hydrological dependency patterns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.events.detectors._shared import (
    clamp,
    safe_quantile,
    select_series_context,
    series_to_float,
    window_size_for_hours,
)
from src.events.rules import SEASONAL_DEPENDENCY_RULE
from src.events.schemas import EventDetection

SEASON_MAP = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}

SEASON_NAMES_PL = {
    "winter": "zima",
    "spring": "wiosna",
    "summer": "lato",
    "autumn": "jesień",
}


def _get_season(timestamp: pd.Timestamp | None) -> str | None:
    if timestamp is None or pd.isna(timestamp):
        return None
    return SEASON_MAP.get(int(timestamp.month))


def _describe_season(season: str | None) -> str:
    if season is None:
        return "nieustalony sezon"
    return SEASON_NAMES_PL.get(season, season)


RAINFALL_COLUMNS = (
    "rainfall_mm",
    "precipitation_mm",
    "rain_mm",
    "rainfall",
    "precip_mm",
)

TEMPERATURE_COLUMNS = (
    "temperature_c",
    "air_temperature_c",
    "water_temperature_c",
    "temp_c",
)

WATER_LEVEL_COLUMNS = (
    "water_level_m",
    "level_m",
    "water_level_cm",
)


def _water_level_signal(
    df: pd.DataFrame, step_hours: float | None
) -> tuple[float, float, int] | None:
    column = next((name for name in WATER_LEVEL_COLUMNS if name in df.columns), None)
    if column is None:
        return None

    series: pd.Series = pd.to_numeric(df[column], errors="coerce")
    if series.dropna().empty:
        return None

    window_rows = window_size_for_hours(72.0, step_hours)
    rolling_trend = series.diff().rolling(window=window_rows, min_periods=1).mean()
    valid_positions = np.flatnonzero(series.notna().to_numpy())
    latest_pos = int(valid_positions[-1])
    latest_trend = (
        float(rolling_trend.iloc[latest_pos]) if pd.notna(rolling_trend.iloc[latest_pos]) else 0.0
    )
    latest_value = float(series.iloc[latest_pos])
    return latest_value, latest_trend, window_rows


def _thaw_like_signal(temperature: pd.Series, current_index: int) -> tuple[float, float, float]:
    current_temp = series_to_float(temperature.iloc[current_index])
    start_index = max(0, current_index - 47)
    previous = temperature.iloc[start_index:current_index]
    if previous.dropna().empty:
        return current_temp, current_temp, 0.0
    min_previous = float(previous.min())
    temp_increase = current_temp - min_previous
    return current_temp, min_previous, temp_increase


def detect_seasonal_dependencies(df: pd.DataFrame) -> EventDetection:
    rainfall_context = select_series_context(df, RAINFALL_COLUMNS)
    temperature_context = select_series_context(df, TEMPERATURE_COLUMNS)

    if rainfall_context is None and temperature_context is None:
        return EventDetection(
            event_type=SEASONAL_DEPENDENCY_RULE.event_type,
            detected=False,
            confidence=0.0,
            severity=0.0,
            metadata={"reason": "missing_meteo_series"},
            message="Brak danych meteorologicznych do oceny zależności sezonowych.",
        )

    base_context = rainfall_context or temperature_context
    assert base_context is not None
    current_season = _get_season(base_context.latest_timestamp)
    current_timestamp = base_context.latest_timestamp
    season_label = _describe_season(current_season)

    rain_score = 0.0
    rain_details: dict[str, object] = {}
    if rainfall_context is not None:
        rainfall = rainfall_context.values.fillna(0.0)
        rain_threshold = max(safe_quantile(rainfall, 0.75, fallback=0.0), 1.0)
        rain_24h = rainfall.rolling(
            window=window_size_for_hours(24.0, rainfall_context.step_hours), min_periods=1
        ).sum()
        rain_7d = rainfall.rolling(
            window=window_size_for_hours(168.0, rainfall_context.step_hours), min_periods=1
        ).sum()
        latest_index = rainfall_context.latest_index
        current_rain = series_to_float(rainfall.iloc[latest_index])
        rain_score = max(
            current_rain / rain_threshold,
            rain_24h.iloc[latest_index] / max(rain_threshold * 3.0, 1e-6),
            rain_7d.iloc[latest_index] / max(rain_threshold * 10.0, 1e-6),
        )
        rain_details = {
            "current_rainfall_mm": current_rain,
            "rainfall_threshold_mm": rain_threshold,
            "rainfall_24h_mm": series_to_float(rain_24h.iloc[latest_index]),
            "rainfall_7d_mm": series_to_float(rain_7d.iloc[latest_index]),
        }

    thaw_score = 0.0
    thaw_details: dict[str, object] = {}
    if temperature_context is not None:
        temperature = temperature_context.values
        current_temp, min_previous, temp_increase = _thaw_like_signal(
            temperature, temperature_context.latest_index
        )
        thaw_threshold = SEASONAL_DEPENDENCY_RULE.thresholds.get("temperature_increase_c", 5.0)
        thaw_mean_threshold = SEASONAL_DEPENDENCY_RULE.thresholds.get("temperature_mean_c", 0.0)

        # ONLY consider thaw if there was freezing AND we are in a thaw-friendly month
        is_thaw_month = current_timestamp is not None and current_timestamp.month in {
            11,
            12,
            1,
            2,
            3,
            4,
        }
        if is_thaw_month and min_previous <= 0.0 and current_temp > 0.0:
            thaw_score = max(
                (current_temp - thaw_mean_threshold) / 2.0,  # dampen the absolute temp effect
                temp_increase / max(thaw_threshold, 1e-6),
            )
        else:
            thaw_score = 0.0
        thaw_details = {
            "current_temperature_c": current_temp,
            "lookback_min_temperature_c": min_previous,
            "temperature_increase_c": temp_increase,
            "season": season_label,
        }

    water_signal = _water_level_signal(df, base_context.step_hours)
    saturation_score = 0.0
    saturation_details: dict[str, object] = {}
    if water_signal is not None:
        water_value, water_trend, window_rows = water_signal
        saturation_score = max(water_value, max(water_trend, 0.0) * 10.0)
        saturation_details = {
            "water_level_value": water_value,
            "water_level_trend": water_trend,
            "trend_window_rows": window_rows,
        }

    season_weights = {
        "winter": {"thaw": 1.2, "rain": 0.8, "saturation": 1.0},
        "spring": {"thaw": 1.15, "rain": 1.0, "saturation": 0.95},
        "summer": {"thaw": 0.65, "rain": 1.3, "saturation": 0.85},
        "autumn": {"thaw": 0.85, "rain": 1.15, "saturation": 1.2},
    }
    weights = season_weights.get(
        current_season or "", {"thaw": 1.0, "rain": 1.0, "saturation": 1.0}
    )
    rain_score *= weights["rain"]
    thaw_score *= weights["thaw"]
    saturation_score *= weights["saturation"]

    candidates = {
        "opadowy": rain_score,
        "roztopowy": thaw_score,
        "nasycenie": saturation_score,
    }
    dominant_factor, dominant_score = max(candidates.items(), key=lambda item: item[1])

    detected = dominant_score >= 1.0 and current_season is not None
    if current_season in {"winter", "spring"}:
        detected = (detected and dominant_factor in {"roztopowy", "nasycenie"}) or thaw_score >= 1.0
    elif current_season in {"summer", "autumn"}:
        detected = (detected and dominant_factor in {"opadowy", "nasycenie"}) or rain_score >= 1.0

    if not detected and max(candidates.values()) >= 1.0:
        detected = True

    latest_timestamp = current_timestamp
    if latest_timestamp is None and base_context.timestamps is not None:
        latest_timestamp = pd.to_datetime(
            base_context.timestamps.iloc[base_context.latest_index], errors="coerce"
        )
        if pd.isna(latest_timestamp):
            latest_timestamp = None

    if not detected:
        return EventDetection(
            event_type=SEASONAL_DEPENDENCY_RULE.event_type,
            detected=False,
            confidence=0.0,
            severity=0.0,
            timestamp=latest_timestamp,
            metadata={
                "season": season_label,
                "dominant_factor": dominant_factor,
                "scores": candidates,
                **rain_details,
                **thaw_details,
                **saturation_details,
            },
            message=(
                f"Brak wyraźnej zależności sezonowej. Sezon: {season_label}; "
                f"dominujący sygnał: {dominant_factor} ({dominant_score:.2f})."
            ),
        )

    confidence = clamp(0.58 + min(max(dominant_score - 1.0, 0.0), 1.5) * 0.18)
    severity = clamp(dominant_score / 2.0)

    if dominant_factor == "roztopowy":
        readable_factor = "roztopowy"
        extra = f"Temperatura wzrosła o {thaw_details.get('temperature_increase_c', 0.0):.1f}°C."
    elif dominant_factor == "nasycenie":
        readable_factor = "nasycenie zlewni"
        extra = f"Trend poziomu wody wskazuje na {saturation_details.get('water_level_trend', 0.0):.3f}."
    else:
        readable_factor = "opadowy"
        extra = f"Opad 24h wynosi {rain_details.get('rainfall_24h_mm', 0.0):.1f} mm."

    return EventDetection(
        event_type=SEASONAL_DEPENDENCY_RULE.event_type,
        detected=True,
        timestamp=latest_timestamp,
        confidence=confidence,
        severity=severity,
        metadata={
            "season": season_label,
            "dominant_factor": readable_factor,
            "scores": candidates,
            **rain_details,
            **thaw_details,
            **saturation_details,
        },
        message=(
            f"{SEASONAL_DEPENDENCY_RULE.response_message} "
            f"Sezon: {season_label}, dominujący mechanizm: {readable_factor}. {extra}"
        ),
    )
