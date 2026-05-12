"""Defines seasonal dependencies that can influence the occurrence of events.
Criteria:
Other events that can influence the occurence of precipitation or increased water levels
Such as:
Summer - heatwaves, droughts, high evaporation rates
Autumn - soil saturation from previous seasons
Winter - snow accumulation, ice formation, thawing episodes
Spring - snowmelt, increased rainfall, thawing episodes, higher plant coverage

Response:
"W tym sezonie dominują inne czynniki podnoszące poziom wody"
+ preferably identifation of the key factor(s)
"""

from __future__ import annotations

import pandas as pd

from src.events.rules import SEASONAL_DEPENDENCY_RULE
from src.events.schemas import EventDetection


def _month_to_season(month: int) -> str:
    # meteorological seasons
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _ensure_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    else:
        df = df.copy()
    return df


def _safe_corr(series_a: pd.Series, series_b: pd.Series) -> float:
    paired = pd.concat([series_a, series_b], axis=1).dropna()
    if len(paired) < 2:
        return 0.0
    if paired.iloc[:, 0].nunique(dropna=True) < 2:
        return 0.0
    if paired.iloc[:, 1].nunique(dropna=True) < 2:
        return 0.0
    value = paired.iloc[:, 0].corr(paired.iloc[:, 1])
    return float(value) if pd.notna(value) else 0.0


def detect_seasonal_dependencies(df: pd.DataFrame) -> EventDetection:
    df_idx = _ensure_ts_index(df)

    detected = False
    confidence = 0.0
    severity = 0.0
    metadata: dict = {}

    # need water level and at least one driver (rain/temperature)
    if "water_level_m" not in df_idx.columns:
        return EventDetection(
            event_type=SEASONAL_DEPENDENCY_RULE.event_type,
            detected=False,
            message=SEASONAL_DEPENDENCY_RULE.response_message,
        )

    # pick current season (based on last timestamp)
    last_ts = pd.to_datetime(df_idx.index[-1])
    month = int(last_ts.month)
    season = _month_to_season(month)

    # compute correlations for season slice
    season_months = {
        "winter": {12, 1, 2},
        "spring": {3, 4, 5},
        "summer": {6, 7, 8},
        "autumn": {9, 10, 11},
    }
    months = pd.DatetimeIndex(df_idx.index).month
    season_mask = months.isin(season_months.get(season, set()))
    season_df = df_idx.loc[season_mask]

    if season_df.empty:
        return EventDetection(
            event_type=SEASONAL_DEPENDENCY_RULE.event_type,
            detected=False,
            message=SEASONAL_DEPENDENCY_RULE.response_message,
        )

    drivers = [
        c for c in ["rainfall_mm", "temperature_c", "wind_speed_ms"] if c in season_df.columns
    ]
    if not drivers:
        return EventDetection(
            event_type=SEASONAL_DEPENDENCY_RULE.event_type,
            detected=False,
            message=SEASONAL_DEPENDENCY_RULE.response_message,
        )

    # compute absolute Pearson correlations with water_level_m
    corrs: list[tuple[str, float]] = []
    for d in drivers:
        corr = _safe_corr(season_df["water_level_m"], season_df[d])
        corrs.append((d, corr))

    # pick dominant driver by absolute correlation
    corrs_sorted = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
    dominant, corr_val = corrs_sorted[0]

    # confidence based on |corr| scaled to [0,1]
    confidence = min(1.0, abs(corr_val))
    severity = confidence * 0.8
    detected = confidence > 0.05

    metadata.update({"season": season, "dominant_factor": dominant, "correlation": float(corr_val)})

    return EventDetection(
        event_type=SEASONAL_DEPENDENCY_RULE.event_type,
        detected=detected,
        timestamp=season_df.index[-1] if detected else None,
        confidence=float(confidence) if detected else None,
        severity=float(severity) if detected else None,
        metadata=metadata,
        message=SEASONAL_DEPENDENCY_RULE.response_message,
    )
