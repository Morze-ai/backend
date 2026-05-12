"""Event detector implementations used by the rule-based sensor system."""

from src.events.detectors.rainfall import detect_flash_flood, detect_long_rainfall
from src.events.detectors.seasonal import detect_seasonal_dependencies
from src.events.detectors.thaw import detect_thaw

__all__ = [
    "detect_flash_flood",
    "detect_long_rainfall",
    "detect_seasonal_dependencies",
    "detect_thaw",
]
