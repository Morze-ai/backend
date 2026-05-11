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

import pandas as pd

from src.events.rules import SEASONAL_DEPENDENCY_RULE
from src.events.schemas import EventDetection


def detect_seasonal_dependencies(df: pd.DataFrame) -> EventDetection:
    detected = False  # TODO: needs implementation

    return EventDetection(
        event_type=SEASONAL_DEPENDENCY_RULE.event_type,
        detected=detected,
        confidence=0.82,
        severity=0.64,
        metadata={
            "temperature_change": 7.1,
        },
        message=SEASONAL_DEPENDENCY_RULE.response_message,
    )
