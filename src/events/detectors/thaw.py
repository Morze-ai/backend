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


def detect_thaw(df: pd.DataFrame) -> EventDetection:
    detected = False  # TODO: needs implementation

    return EventDetection(
        event_type=THAW_RULE.event_type,
        detected=detected,
        confidence=0.82,
        severity=0.64,
        metadata={
            "temperature_change": 7.1,
        },
        message=THAW_RULE.response_message,
    )
