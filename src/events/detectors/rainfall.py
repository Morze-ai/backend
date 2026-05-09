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


def detect_long_rainfall(df: pd.DataFrame) -> EventDetection:
    detected = False  # TODO: needs implementation

    return EventDetection(
        event_type=LONG_RAINFALL_RULE.event_type,
        detected=detected,
        confidence=0.82,
        severity=0.64,
        metadata={
            "temperature_change": 7.1,
        },
        message=LONG_RAINFALL_RULE.response_message,
    )


def detect_flash_flood(df: pd.DataFrame) -> EventDetection:
    detected = False  # TODO: needs implementation

    return EventDetection(
        event_type=FLASH_FLOOD_RULE.event_type,
        detected=detected,
        confidence=0.82,
        severity=0.64,
        metadata={
            "temperature_change": 7.1,
        },
        message=FLASH_FLOOD_RULE.response_message,
    )
