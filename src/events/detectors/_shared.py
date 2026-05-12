"""Shared utilities for rule-based event detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_TIMESTAMP_CANDIDATES = ("timestamp", "date", "datetime")


@dataclass(slots=True)
class SeriesContext:
    """Convenience container for a detector's selected time series."""

    values: pd.Series
    timestamps: pd.Series | None
    step_hours: float | None
    latest_index: int
    latest_timestamp: pd.Timestamp | None


def find_first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first available column from a list of candidates."""

    for column in candidates:
        if column in df.columns:
            return column
    return None


def numeric_series(
    df: pd.DataFrame, candidates: tuple[str, ...]
) -> tuple[str | None, pd.Series | None]:
    """Return the first numeric series matching the provided candidates."""

    column = find_first_column(df, candidates)
    if column is None:
        return None, None

    series: pd.Series = pd.to_numeric(df[column], errors="coerce")
    return column, series


def datetime_series(
    df: pd.DataFrame, candidates: tuple[str, ...] = DEFAULT_TIMESTAMP_CANDIDATES
) -> tuple[str | None, pd.Series | None]:
    """Return the first datetime-like series matching the provided candidates."""

    column = find_first_column(df, candidates)
    if column is None:
        return None, None

    series = pd.to_datetime(df[column], errors="coerce")
    return column, series


def infer_step_hours(timestamps: pd.Series | None) -> float | None:
    """Infer the median sampling step in hours from a timestamp series."""

    if timestamps is None:
        return None

    valid = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    if len(valid) < 2:
        return None

    deltas = valid.diff().dropna().dt.total_seconds() / 3600.0
    positive = deltas[deltas > 0]
    if positive.empty:
        return None

    step_hours = float(positive.median())
    if not np.isfinite(step_hours) or step_hours <= 0:
        return None
    return step_hours


def window_size_for_hours(hours: float, step_hours: float | None) -> int:
    """Translate a time window in hours into a rolling row window."""

    if step_hours is None or step_hours <= 0:
        return max(1, round(hours))
    return max(1, round(hours / step_hours))


def latest_valid_index(values: pd.Series) -> int | None:
    """Return the last index containing a valid observation."""

    valid_positions = np.flatnonzero(values.notna().to_numpy())
    if valid_positions.size == 0:
        return None
    return int(valid_positions[-1])


def select_series_context(df: pd.DataFrame, candidates: tuple[str, ...]) -> SeriesContext | None:
    """Build a common detector context for a numeric series and optional timestamps."""

    _, values = numeric_series(df, candidates)
    if values is None:
        return None

    _, timestamps = datetime_series(df)
    step_hours = infer_step_hours(timestamps)
    index = latest_valid_index(values)
    if index is None:
        return None

    latest_timestamp = None
    if timestamps is not None:
        latest_timestamp = pd.to_datetime(timestamps.iloc[index], errors="coerce")
        if pd.isna(latest_timestamp):
            latest_timestamp = None

    return SeriesContext(
        values=values,
        timestamps=timestamps,
        step_hours=step_hours,
        latest_index=index,
        latest_timestamp=latest_timestamp,
    )


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp a numeric value to the requested range."""

    return float(max(minimum, min(maximum, value)))


def safe_quantile(values: pd.Series, quantile: float, fallback: float = 0.0) -> float:
    """Compute a quantile on valid values with a safe fallback."""

    numeric_values: pd.Series = pd.to_numeric(values, errors="coerce")
    valid = numeric_values.dropna()
    if valid.empty:
        return fallback
    result = float(valid.quantile(quantile))
    if not np.isfinite(result):
        return fallback
    return result


def series_to_float(value: Any, fallback: float = 0.0) -> float:
    """Convert a possibly missing scalar to a float."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(result):
        return fallback
    return result
