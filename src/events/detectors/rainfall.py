"""Detects rainfall episodes such as long saturation periods and flash floods."""

from __future__ import annotations

import pandas as pd

from src.events.detectors._shared import (
    clamp,
    safe_quantile,
    select_series_context,
    series_to_float,
    window_size_for_hours,
)
from src.events.rules import FLASH_FLOOD_RULE, LONG_RAINFALL_RULE
from src.events.schemas import EventDetection

RAINFALL_COLUMNS = (
    "rainfall_mm",
    "precipitation_mm",
    "rain_mm",
    "rainfall",
    "precip_mm",
)


def _build_message(base_message: str, details: list[str], detected: bool) -> str:
    if not detected:
        if details:
            return f"Brak sygnału opadowego. {details[0]}"
        return "Brak sygnału opadowego w dostępnych danych."
    if details:
        return f"{base_message} {' '.join(details)}"
    return base_message


def _episode_metadata(
    *,
    timestamp: pd.Timestamp | None,
    rainfall_now: float,
    rainfall_72h: float,
    rainfall_7d: float,
    window_rows_24h: int,
    score: float,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "rainfall_mm": rainfall_now,
        "rainfall_72h_mm": rainfall_72h,
        "rainfall_7d_mm": rainfall_7d,
        "window_rows_24h": window_rows_24h,
        "score": score,
    }


def detect_long_rainfall(df: pd.DataFrame) -> EventDetection:
    context = select_series_context(df, RAINFALL_COLUMNS)
    if context is None:
        return EventDetection(
            event_type=LONG_RAINFALL_RULE.event_type,
            detected=False,
            confidence=0.0,
            severity=0.0,
            metadata={"reason": "missing_rainfall_series"},
            message="Brak kolumny z opadem do wykrywania długotrwałych opadów.",
        )

    rainfall = context.values.fillna(0.0)
    step_hours = context.step_hours
    window_72h = window_size_for_hours(72.0, step_hours)
    window_7d = window_size_for_hours(168.0, step_hours)
    window_24h = window_size_for_hours(24.0, step_hours)

    rolling_72h = rainfall.rolling(window=window_72h, min_periods=1).sum()
    rolling_7d = rainfall.rolling(window=window_7d, min_periods=1).sum()

    current_threshold = LONG_RAINFALL_RULE.thresholds["current_rainfall_mm_h"]
    rainfall_72h_threshold = LONG_RAINFALL_RULE.thresholds["rainfall_72h_mm"]
    rainfall_7d_threshold = LONG_RAINFALL_RULE.thresholds["rainfall_7d_mm"]

    valid_mask = (
        rainfall.ge(current_threshold)
        & rolling_72h.ge(rainfall_72h_threshold)
        & rolling_7d.ge(rainfall_7d_threshold)
    )

    if not bool(valid_mask.any()):
        latest_timestamp = None
        if context.timestamps is not None:
            latest_timestamp = pd.to_datetime(
                context.timestamps.iloc[context.latest_index], errors="coerce"
            )
        return EventDetection(
            event_type=LONG_RAINFALL_RULE.event_type,
            detected=False,
            confidence=0.0,
            severity=0.0,
            metadata={
                "reason": "thresholds_not_met",
                "latest_timestamp": latest_timestamp,
                "current_rainfall_mm": series_to_float(rainfall.iloc[context.latest_index]),
            },
            message=_build_message(
                LONG_RAINFALL_RULE.response_message,
                ["Nie przekroczono progów 72h/7d ani bieżącej intensywności opadu."],
                detected=False,
            ),
        )

    score_72h = rolling_72h / max(rainfall_72h_threshold, 1e-6)
    score_7d = rolling_7d / max(rainfall_7d_threshold, 1e-6)
    score_now = rainfall / max(current_threshold, 1e-6)
    combined_score = pd.concat([score_72h, score_7d, score_now], axis=1).max(axis=1)
    best_pos = int(combined_score.to_numpy().argmax())
    best_score = float(combined_score.iloc[best_pos])
    rainfall_now = series_to_float(rainfall.iloc[best_pos])
    rainfall_72h = series_to_float(rolling_72h.iloc[best_pos])
    rainfall_7d = series_to_float(rolling_7d.iloc[best_pos])
    latest_timestamp = None
    if context.timestamps is not None:
        latest_timestamp = pd.to_datetime(context.timestamps.iloc[best_pos], errors="coerce")
        if pd.isna(latest_timestamp):
            latest_timestamp = None

    confidence = clamp(0.55 + min(max(best_score - 1.0, 0.0), 1.5) * 0.25)
    severity = clamp(
        max(score_72h.iloc[best_pos], score_7d.iloc[best_pos], score_now.iloc[best_pos]) / 2.0
    )

    message = _build_message(
        LONG_RAINFALL_RULE.response_message,
        [
            f"Najsilniejszy epizod wystąpił {latest_timestamp}.",
            f"Opad bieżący: {rainfall_now:.1f} mm.",
            f"Suma 72h: {rainfall_72h:.1f} mm.",
            f"Suma 7d: {rainfall_7d:.1f} mm.",
        ],
        detected=True,
    )

    return EventDetection(
        event_type=LONG_RAINFALL_RULE.event_type,
        detected=True,
        timestamp=latest_timestamp,
        confidence=confidence,
        severity=severity,
        metadata=_episode_metadata(
            timestamp=latest_timestamp,
            rainfall_now=rainfall_now,
            rainfall_72h=rainfall_72h,
            rainfall_7d=rainfall_7d,
            window_rows_24h=window_24h,
            score=best_score,
        ),
        message=message,
    )


def detect_flash_flood(df: pd.DataFrame) -> EventDetection:
    context = select_series_context(df, RAINFALL_COLUMNS)
    if context is None:
        return EventDetection(
            event_type=FLASH_FLOOD_RULE.event_type,
            detected=False,
            confidence=0.0,
            severity=0.0,
            metadata={"reason": "missing_rainfall_series"},
            message="Brak kolumny z opadem do wykrywania epizodu gwałtownego opadu.",
        )

    rainfall = context.values.fillna(0.0)
    step_hours = context.step_hours
    window_hours = FLASH_FLOOD_RULE.thresholds["duration_hours"]
    window_rows = window_size_for_hours(window_hours, step_hours)
    positive_count = rainfall.gt(0.0).rolling(window=window_rows, min_periods=1).sum()
    rolling_sum = rainfall.rolling(window=window_rows, min_periods=1).sum()

    positive_values = rainfall[rainfall > 0.0]
    percentile_threshold = safe_quantile(
        positive_values if not positive_values.empty else rainfall,
        FLASH_FLOOD_RULE.thresholds["rainfall_percentile"] / 100.0,
        fallback=max(float(rainfall.max()) * 0.75, 0.1),
    )
    if percentile_threshold <= 0.0:
        percentile_threshold = max(float(rainfall.max()) * 0.75, 0.1)

    sustained_threshold = max(1, round(window_rows * 0.5))
    valid_mask = rainfall.ge(percentile_threshold) & positive_count.ge(sustained_threshold)

    if not bool(valid_mask.any()):
        latest_timestamp = None
        if context.timestamps is not None:
            latest_timestamp = pd.to_datetime(
                context.timestamps.iloc[context.latest_index], errors="coerce"
            )
        return EventDetection(
            event_type=FLASH_FLOOD_RULE.event_type,
            detected=False,
            confidence=0.0,
            severity=0.0,
            metadata={
                "reason": "thresholds_not_met",
                "latest_timestamp": latest_timestamp,
                "percentile_threshold_mm": percentile_threshold,
            },
            message=_build_message(
                FLASH_FLOOD_RULE.response_message,
                [
                    "Nie wykryto pojedynczego bardzo intensywnego opadu z wystarczająco długim utrzymaniem.",
                ],
                detected=False,
            ),
        )

    score_intensity = rainfall / max(percentile_threshold, 1e-6)
    score_duration = positive_count / max(float(sustained_threshold), 1.0)
    score_volume = rolling_sum / max(percentile_threshold * max(window_rows * 0.25, 1.0), 1e-6)
    combined_score = pd.concat([score_intensity, score_duration, score_volume], axis=1).max(axis=1)
    best_pos = int(combined_score.to_numpy().argmax())
    best_score = float(combined_score.iloc[best_pos])
    latest_timestamp = None
    if context.timestamps is not None:
        latest_timestamp = pd.to_datetime(context.timestamps.iloc[best_pos], errors="coerce")
        if pd.isna(latest_timestamp):
            latest_timestamp = None

    current_rain = series_to_float(rainfall.iloc[best_pos])
    current_sum = series_to_float(rolling_sum.iloc[best_pos])
    current_count = series_to_float(positive_count.iloc[best_pos])

    confidence = clamp(0.60 + min(max(best_score - 1.0, 0.0), 1.5) * 0.20)
    severity = clamp(
        max(
            score_intensity.iloc[best_pos],
            score_duration.iloc[best_pos],
            score_volume.iloc[best_pos],
        )
        / 2.0
    )

    message = _build_message(
        FLASH_FLOOD_RULE.response_message,
        [
            f"Najsilniejszy sygnał: {latest_timestamp}.",
            f"Bieżący opad: {current_rain:.1f} mm.",
            f"Suma w oknie {window_hours:.0f}h: {current_sum:.1f} mm.",
            f"Dodatnie obserwacje w oknie: {int(current_count)}/{window_rows}.",
        ],
        detected=True,
    )

    return EventDetection(
        event_type=FLASH_FLOOD_RULE.event_type,
        detected=True,
        timestamp=latest_timestamp,
        confidence=confidence,
        severity=severity,
        metadata={
            "timestamp": latest_timestamp,
            "rainfall_mm": current_rain,
            "rolling_sum_mm": current_sum,
            "positive_count": int(current_count),
            "percentile_threshold_mm": percentile_threshold,
            "window_rows": window_rows,
            "score": best_score,
        },
        message=message,
    )
