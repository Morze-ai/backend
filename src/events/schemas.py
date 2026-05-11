"""Defines the data structures for representing event rules, detections, and evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class StrEnum(str, Enum):  # noqa: UP042
    """Fallback StrEnum for Python versions before 3.11."""


class EventType(StrEnum):
    """Supported hydrological event categories."""

    LONG_RAINFALL = "long_rainfall"
    FLASH_FLOOD = "flash_flood"
    THAW = "thaw"
    SEASONAL_DEPENDENCY = "seasonal_dependency"


@dataclass(slots=True)
class EventRule:
    """Declarative description of an event rule."""

    event_type: EventType
    name: str
    description: str
    thresholds: dict[str, float] = field(default_factory=dict)
    response_message: str = ""


@dataclass(slots=True)
class EventDetection:
    """Standardized detector output."""

    event_type: EventType
    detected: bool
    timestamp: pd.Timestamp | None = None
    confidence: float | None = None
    severity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass(slots=True)
class EventEvaluation:
    """Evaluation summary comparing predictions."""

    event_type: EventType
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
