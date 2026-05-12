"""Logic for attributing events to factors (O1-O4) using SHAP values and rule co-occurrence."""

from __future__ import annotations

from src.events.rules import EventType


def attribute_event_type(
    top_features: list[str],
) -> EventType:
    """
    Determines the most likely event type (O1-O4) based on the most influential features.
    """

    feature_sets = {
        EventType.FLASH_FLOOD: ["rainfall_mm", "rain_1h_sum", "rain_3h_sum"],
        EventType.LONG_RAINFALL: ["rain_24h_sum", "rain_lag_24h", "soil_saturation_index"],
        EventType.THAW: ["temperature_c", "temp_mean", "thaw_flag", "temp_delta_24h"],
        EventType.SEASONAL_DEPENDENCY: ["month", "day_of_year", "season"],
    }

    scores = {etype: 0.0 for etype in EventType}

    for i, feature in enumerate(top_features):
        # Weighted score: top features contribute more
        weight = 1.0 / (i + 1)
        for etype, features in feature_sets.items():
            if any(f in feature for f in features):
                scores[etype] += weight

    # Default to flash flood if no strong signal, otherwise pick max
    if all(s == 0 for s in scores.values()):
        return EventType.FLASH_FLOOD

    return max(scores, key=lambda k: scores[k])


def compute_historical_confidence(etype: EventType, output_dir: str | None = None) -> float:
    """
    Estimates confidence by checking how often this event type historically
    co-occurred with high water levels.
    """

    # Try to load from calculated JSON first
    if output_dir:
        import json
        from pathlib import Path

        conf_path = Path(output_dir) / "historical_confidence.json"
        if conf_path.exists():
            try:
                with conf_path.open() as f:
                    confidences = json.load(f)
                    return confidences.get(etype.value, 0.5)
            except Exception:
                pass

    reliability_map = {
        EventType.FLASH_FLOOD: 0.85,
        EventType.LONG_RAINFALL: 0.75,
        EventType.THAW: 0.70,
        EventType.SEASONAL_DEPENDENCY: 0.60,
    }

    return reliability_map.get(etype, 0.5)
