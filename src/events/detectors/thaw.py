"""Describes thawing episodes that may increase water levels due to melting snow and ice.
Criteria:
consistent high temperature (or sudden increase) in winter-early spring season
+ lack/low precipitation (snow) in the preceding period or rainfall on snow
(if rain on snow can be identified)

Response:
"Warunki roztopowe mogą powodować wzrost poziomu wody"
"""

import pandas as pd

from src.events.rules import THAW_RULE
from src.events.schemas import EventDetection


def _ensure_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    else:
        df = df.copy()
    return df


def detect_thaw(df: pd.DataFrame) -> EventDetection:
    df_idx = _ensure_ts_index(df)

    temp_col = "temperature_c"
    wind_col = "wind_speed_ms"

    detected = False
    confidence = 0.0
    severity = 0.0
    metadata: dict = {}

    if temp_col not in df_idx.columns:
        return EventDetection(
            event_type=THAW_RULE.event_type, detected=False, message=THAW_RULE.response_message
        )

    # Recent 24h mean and prior 7d mean
    mean24 = float(df_idx[temp_col].rolling("24h").mean().iloc[-1])
    prior7d_mean = (
        float(df_idx[temp_col].rolling("7D").mean().shift(24).iloc[-1])
        if len(df_idx) > 24
        else float(df_idx[temp_col].rolling("7D").mean().iloc[0])
    )

    t_increase_thr = float(THAW_RULE.thresholds.get("temperature_increase_c", 5.0))
    t_mean_thr = float(THAW_RULE.thresholds.get("temperature_mean_c", 0.0))

    # detect thaw: recent mean > 0 and prior 7d mean <= 0, or recent jump > threshold
    recent_min = float(df_idx[temp_col].rolling("24h").min().iloc[-1])
    recent_max = float(df_idx[temp_col].rolling("24h").max().iloc[-1])
    temp_change = recent_max - recent_min

    if (mean24 > t_mean_thr and prior7d_mean <= 0.0) or (
        temp_change >= t_increase_thr and mean24 > t_mean_thr
    ):
        detected = True
        # confidence proportional to mean above zero and change
        conf_by_mean = min(1.0, max(0.0, (mean24 - t_mean_thr) / (abs(t_mean_thr) + 5.0)))
        conf_by_change = min(1.0, temp_change / (t_increase_thr + 1e-9))
        confidence = max(conf_by_mean, conf_by_change)
        severity = confidence

    metadata["mean24_c"] = mean24
    metadata["prior7d_mean_c"] = prior7d_mean
    metadata["temp_change_24h_c"] = temp_change

    # wind context: if strong wind present, increase severity slightly (evaporation/transport)
    if wind_col in df_idx.columns and not df_idx[wind_col].dropna().empty:
        wind_speed = float(df_idx[wind_col].iloc[-1])
        metadata["wind_speed_ms"] = wind_speed
        if wind_speed >= 6.0 and detected:
            confidence = min(1.0, confidence + 0.1)
            severity = min(1.0, severity + 0.1)

    return EventDetection(
        event_type=THAW_RULE.event_type,
        detected=detected,
        timestamp=df_idx.index[-1] if detected else None,
        confidence=float(confidence) if detected else None,
        severity=float(severity) if detected else None,
        metadata=metadata,
        message=THAW_RULE.response_message,
    )
