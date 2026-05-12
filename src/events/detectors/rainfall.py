"""Describes episodes of rainfall. Checks for two types of rain related events - long saturation episodes or flash floods.

Long rainfall saturation episodes:
Criteria:
Cumulative rainfall 72h / 7 days against seasonal threshold + current rainfall > threshold

Response:
"Zlewnia jest nasycona - nawet umiarkowany opad może podnieść poziom
wody."

Flash flood episodes:
Criteria:
Intensity > 90%
and/or
Duration > 24h against seasonal threshiold

Response:
"Wysoka intensywność opadu sprzyja gwałtownemu wzrostowi poziomu
wody."
"""

import pandas as pd

from src.events.rules import FLASH_FLOOD_RULE, LONG_RAINFALL_RULE
from src.events.schemas import EventDetection


def _ensure_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    else:
        df = df.copy()
    return df


def detect_long_rainfall(df: pd.DataFrame) -> EventDetection:
    df_idx = _ensure_ts_index(df)

    # Required series
    rain_col = "rainfall_mm"
    sat_col = "soil_saturation_index"

    detected = False
    confidence = 0.0
    severity = 0.0
    metadata: dict = {}

    if rain_col not in df_idx.columns:
        return EventDetection(
            event_type=LONG_RAINFALL_RULE.event_type,
            detected=False,
            message=LONG_RAINFALL_RULE.response_message,
        )

    # rolling sums
    r72 = df_idx[rain_col].rolling("72h").sum()
    r7d = df_idx[rain_col].rolling("7D").sum()

    last_idx = r72.index[-1]
    last_72 = float(r72.iloc[-1]) if not r72.empty else 0.0
    last_7d = float(r7d.iloc[-1]) if not r7d.empty else 0.0

    t72_thr = float(LONG_RAINFALL_RULE.thresholds.get("rainfall_72h_mm", 40.0))
    t7d_thr = float(LONG_RAINFALL_RULE.thresholds.get("rainfall_7d_mm", 90.0))
    current_h = float(df_idx[rain_col].iloc[-1])

    # basic detection: both cumulative windows exceed thresholds or current rainfall high
    if (last_72 >= t72_thr and last_7d >= t7d_thr) or (
        current_h >= float(LONG_RAINFALL_RULE.thresholds.get("current_rainfall_mm_h", 4.0))
        and last_72 >= t72_thr
    ):
        detected = True
        # confidence based on how far above thresholds
        conf72 = min(1.0, last_72 / (t72_thr + 1e-9))
        conf7d = min(1.0, last_7d / (t7d_thr + 1e-9))
        confidence = max(conf72, conf7d)
        severity = confidence

    # boost or reduce confidence with soil saturation if available
    if sat_col in df_idx.columns and not df_idx[sat_col].dropna().empty:
        sat = float(df_idx[sat_col].iloc[-1])
        metadata["soil_saturation_index"] = sat
        # if saturation high (>0.7) increase confidence
        if sat >= 0.7:
            confidence = min(1.0, confidence + 0.15)
            severity = min(1.0, severity + 0.15)

    return EventDetection(
        event_type=LONG_RAINFALL_RULE.event_type,
        detected=detected,
        timestamp=last_idx if detected else None,
        confidence=float(confidence) if detected else None,
        severity=float(severity) if detected else None,
        metadata=metadata,
        message=LONG_RAINFALL_RULE.response_message,
    )


def detect_flash_flood(df: pd.DataFrame) -> EventDetection:
    df_idx = _ensure_ts_index(df)

    rain_col = "rainfall_mm"
    detected = False
    confidence = 0.0
    severity = 0.0
    metadata: dict = {}

    if rain_col not in df_idx.columns:
        return EventDetection(
            event_type=FLASH_FLOOD_RULE.event_type,
            detected=False,
            message=FLASH_FLOOD_RULE.response_message,
        )

    # compute 6h rolling sum and historical percentile
    r6 = df_idx[rain_col].rolling("6h").sum()
    r24 = df_idx[rain_col].rolling("24h").sum()

    # historical percentile for 6h sums
    try:
        perc = float(FLASH_FLOOD_RULE.thresholds.get("rainfall_percentile", 90.0))
        hist_p90 = float(r6.dropna().quantile(perc / 100.0)) if not r6.dropna().empty else 0.0
    except Exception:
        hist_p90 = 0.0

    last_idx = r6.index[-1]
    last_6 = float(r6.iloc[-1]) if not r6.empty else 0.0
    last_24 = float(r24.iloc[-1]) if not r24.empty else 0.0

    # simple rule: 6h sum exceeds historical percentile -> flash flood condition
    if hist_p90 > 0 and last_6 >= hist_p90:
        detected = True
        confidence = min(1.0, last_6 / (hist_p90 + 1e-9))
        severity = confidence

    # duration-based heuristic: sustained high rainfall over configured duration
    # if 24h cumulative rainfall is very high relative to its median, also trigger
    if not detected:
        med24 = float(r24.dropna().median()) if not r24.dropna().empty else 0.0
        if med24 > 0 and last_24 >= med24 * 1.5 and last_24 > 0:
            detected = True
            confidence = min(1.0, last_24 / (med24 * 2.0 + 1e-9))
            severity = confidence * 0.9

    metadata.update({"6h_mm": last_6, "24h_mm": last_24, "hist_6h_p90": hist_p90})

    return EventDetection(
        event_type=FLASH_FLOOD_RULE.event_type,
        detected=detected,
        timestamp=last_idx if detected else None,
        confidence=float(confidence) if detected else None,
        severity=float(severity) if detected else None,
        metadata=metadata,
        message=FLASH_FLOOD_RULE.response_message,
    )
