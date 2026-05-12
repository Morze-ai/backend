"""Rule-based event detection utilities for hydrological episodes."""

from src.events.detectors import (
    detect_flash_flood,
    detect_long_rainfall,
    detect_seasonal_dependencies,
    detect_thaw,
)
from src.events.rules import EVENT_RULES
from src.events.schemas import EventDetection, EventEvaluation, EventRule, EventType

__all__ = [
    "EVENT_RULES",
    "EventDetection",
    "EventEvaluation",
    "EventRule",
    "EventType",
    "detect_flash_flood",
    "detect_long_rainfall",
    "detect_seasonal_dependencies",
    "detect_thaw",
]
