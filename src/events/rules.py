"""Defines the rules for detecting different types of events based on specific criteria.

Might need updates as we learn more about the data and the events we want to detect.
As well as parameter adjustments
"""

from __future__ import annotations

from src.events.schemas import EventRule, EventType

LONG_RAINFALL_RULE = EventRule(
    event_type=EventType.LONG_RAINFALL,
    name="Long Rainfall Saturation",
    description=(
        "Detects prolonged rainfall saturation episodes based on "
        "cumulative rainfall against seasonal thresholds."
    ),
    thresholds={
        "rainfall_72h_mm": 40.0,
        "rainfall_7d_mm": 90.0,
        "current_rainfall_mm_h": 4.0,
    },
    response_message=("Zlewnia jest nasycona - nawet umiarkowany opad może podnieść poziom wody."),
)


FLASH_FLOOD_RULE = EventRule(
    event_type=EventType.FLASH_FLOOD,
    name="Flash Flood Episode",
    description=("Detects high-intensity rainfall episodes that may rapidly increase water level."),
    thresholds={
        "rainfall_percentile": 90.0,
        "duration_hours": 24.0,
    },
    response_message=("Wysoka intensywność opadu sprzyja gwałtownemu wzrostowi poziomu wody."),
)


THAW_RULE = EventRule(
    event_type=EventType.THAW,
    name="Thaw Episode",
    description=("Detects thawing conditions that may contribute to rising water levels."),
    thresholds={
        "temperature_increase_c": 5.0,
        "temperature_mean_c": 0.0,
    },
    response_message=("Warunki roztopowe mogą powodować wzrost poziomu wody."),
)


SEASONAL_DEPENDENCY_RULE = EventRule(
    event_type=EventType.SEASONAL_DEPENDENCY,
    name="Seasonal Dependency",
    description=("Represents seasonal environmental conditions influencing water level behaviour."),
    thresholds={},
    response_message=("W tym sezonie dominują inne czynniki podnoszące poziom wody."),
)


EVENT_RULES: dict[EventType, EventRule] = {
    rule.event_type: rule
    for rule in [
        LONG_RAINFALL_RULE,
        FLASH_FLOOD_RULE,
        THAW_RULE,
        SEASONAL_DEPENDENCY_RULE,
    ]
}
