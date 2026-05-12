"""Detects thaw episodes that may increase water levels due to snowmelt."""

from __future__ import annotations

import pandas as pd

from src.events.detectors._shared import (
    clamp,
    safe_quantile,
    select_series_context,
    series_to_float,
    window_size_for_hours,
)
from src.events.rules import THAW_RULE
from src.events.schemas import EventDetection

TEMPERATURE_COLUMNS = (
    "temperature_c",
    "air_temperature_c",
    "water_temperature_c",
    "temp_c",
)

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


def _season_is_thaw_friendly(season: str | None, timestamp: pd.Timestamp | None) -> bool:
    if season in {"winter", "spring"}:
        return True
    if timestamp is None or pd.isna(timestamp):
        return False
    return timestamp.month in {11, 12, 1, 2, 3, 4}


def detect_thaw(df: pd.DataFrame) -> EventDetection:
    context = select_series_context(df, TEMPERATURE_COLUMNS)
    if context is None:
        return EventDetection(
            event_type=THAW_RULE.event_type,
            detected=False,
            confidence=0.0,
            severity=0.0,
            metadata={"reason": "missing_temperature_series"},
            message="Brak kolumny z temperaturą do wykrywania roztopów.",
        )

    temperature = context.values
    step_hours = context.step_hours
    lookback_rows = window_size_for_hours(168.0, step_hours)
    current_index = context.latest_index
    start_index = max(0, current_index - lookback_rows + 1)
    lookback = temperature.iloc[start_index : current_index + 1]
    previous = temperature.iloc[start_index:current_index]

    current_temp = series_to_float(temperature.iloc[current_index])
    lookback_mean = safe_quantile(lookback, 0.5, fallback=current_temp)
    min_previous = float(previous.min()) if not previous.dropna().empty else current_temp
    temp_increase = current_temp - min_previous

    mean_threshold = THAW_RULE.thresholds["temperature_mean_c"]
    increase_threshold = THAW_RULE.thresholds["temperature_increase_c"]

    current_season = _get_season(context.latest_timestamp)
    season_ok = _season_is_thaw_friendly(current_season, context.latest_timestamp)

    detected = (
        season_ok
        and current_temp > mean_threshold
        and min_previous <= 0.0
        and temp_increase >= increase_threshold
    )

    latest_timestamp = context.latest_timestamp

    season_label = _describe_season(current_season)
    if not detected:
        return EventDetection(
            event_type=THAW_RULE.event_type,
            detected=False,
            confidence=0.0,
            severity=0.0,
            timestamp=latest_timestamp,
            metadata={
                "season": season_label,
                "current_temperature_c": current_temp,
                "lookback_min_temperature_c": min_previous,
                "lookback_mean_temperature_c": lookback_mean,
                "temperature_increase_c": temp_increase,
                "lookback_rows": lookback_rows,
            },
            message=(
                "Brak sygnału roztopowego. "
                f"Sezon: {season_label}, temperatura bieżąca: {current_temp:.1f}°C, "
                f"wzrost względem minimum: {temp_increase:.1f}°C."
            ),
        )

    positive_margin = max(current_temp - mean_threshold, 0.0)
    confidence = clamp(0.60 + min(max(temp_increase - increase_threshold, 0.0), 8.0) * 0.05)
    severity = clamp(
        max(temp_increase / max(increase_threshold, 1e-6), positive_margin + 0.2) / 2.0
    )

    return EventDetection(
        event_type=THAW_RULE.event_type,
        detected=True,
        timestamp=latest_timestamp,
        confidence=confidence,
        severity=severity,
        metadata={
            "season": season_label,
            "current_temperature_c": current_temp,
            "lookback_min_temperature_c": min_previous,
            "lookback_mean_temperature_c": lookback_mean,
            "temperature_increase_c": temp_increase,
            "lookback_rows": lookback_rows,
        },
        message=(
            f"{THAW_RULE.response_message} "
            f"Sezon: {season_label}, temperatura wzrosła o {temp_increase:.1f}°C, "
            f"a minimum w oknie wynosiło {min_previous:.1f}°C."
        ),
    )
