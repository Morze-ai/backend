"""Evaluates rule-based event detectors and compares them against labels or model predictions."""

# pyright: reportArgumentType=false

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.events.detectors.rainfall import detect_flash_flood, detect_long_rainfall
from src.events.detectors.seasonal import detect_seasonal_dependencies
from src.events.detectors.thaw import detect_thaw
from src.events.schemas import EventDetection, EventEvaluation, EventType

DetectorFn = Callable[[pd.DataFrame], EventDetection]

DEFAULT_EVENT_DETECTORS: dict[EventType, DetectorFn] = {
    EventType.LONG_RAINFALL: detect_long_rainfall,
    EventType.FLASH_FLOOD: detect_flash_flood,
    EventType.THAW: detect_thaw,
    EventType.SEASONAL_DEPENDENCY: detect_seasonal_dependencies,
}


class EventEvaluator:
    """
    Evaluates rule-based event detectors and compares them
    against labels or model predictions.
    """

    def __init__(
        self,
        detectors: dict[EventType, DetectorFn],
    ) -> None:
        self.detectors = detectors

    def run_detectors(
        self,
        df: pd.DataFrame,
    ) -> list[EventDetection]:
        """
        Executes all configured event detectors.
        """

        detections: list[EventDetection] = []

        for detector in self.detectors.values():
            result = detector(df)
            detections.append(result)

        return detections


def build_detector_output_frame(
    frame: pd.DataFrame,
    detectors: dict[EventType, DetectorFn] | None = None,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Evaluate rule detectors on each time prefix and return a per-row output frame."""

    if frame.empty:
        return pd.DataFrame(index=frame.index)

    detector_map = detectors or DEFAULT_EVENT_DETECTORS
    if timestamp_column not in frame.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' is missing in frame.")

    ordered = frame.copy()
    ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column], errors="coerce")
    if ordered[timestamp_column].isna().any():
        raise ValueError(f"Timestamp column '{timestamp_column}' contains invalid values.")
    ordered = ordered.sort_values(timestamp_column).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    detector_items = list(detector_map.items())

    # Pre-sort and set index once to avoid $O(N^2)$ sorting in detectors
    ordered = ordered.set_index(timestamp_column)

    for end_index in tqdm(range(len(ordered)), desc="Event detection", unit="row"):
        # Use a view instead of a full copy where possible
        prefix = ordered.iloc[: end_index + 1]
        row: dict[str, Any] = {}
        active_detections: list[EventDetection] = []

        for event_type, detector in detector_items:
            detection = detector(prefix)
            column_prefix = event_type.value
            row[f"{column_prefix}_detected"] = detection.detected
            row[f"{column_prefix}_confidence"] = detection.confidence
            row[f"{column_prefix}_severity"] = detection.severity
            row[f"{column_prefix}_message"] = detection.message
            row[f"{column_prefix}_metadata"] = detection.metadata
            if detection.detected:
                active_detections.append(detection)

        if active_detections:
            best_detection = max(
                active_detections,
                key=lambda item: (
                    float(item.confidence or 0.0),
                    float(item.severity or 0.0),
                ),
            )
            row["event_type"] = best_detection.event_type.value
            row["detection_confidence"] = best_detection.confidence
            row["detection_severity"] = best_detection.severity
            row["detection_method"] = best_detection.event_type.value
            row["detection_message"] = best_detection.message
            row["contributing_factors"] = ", ".join(sorted(best_detection.metadata.keys()))
        else:
            row["event_type"] = "none"
            row["detection_confidence"] = None
            row["detection_severity"] = None
            row["detection_method"] = None
            row["detection_message"] = ""
            row["contributing_factors"] = ""

        rows.append(row)

    return pd.DataFrame(rows, index=ordered.index)

    def compare_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        event_type: EventType,
    ) -> EventEvaluation:
        """
        Compares predictions against ground truth.
        """

        precision = precision_score(
            # pyright: ignore[reportArgumentType]
            y_true,
            y_pred,
            zero_division=0,
        )
        recall = recall_score(
            # pyright: ignore[reportArgumentType]
            y_true,
            y_pred,
            zero_division=0,
        )
        f1 = f1_score(
            # pyright: ignore[reportArgumentType]
            y_true,
            y_pred,
            zero_division=0,
        )
        accuracy = accuracy_score(y_true, y_pred)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        return EventEvaluation(
            event_type=event_type,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            accuracy=float(accuracy),
        )

    def compare_rule_vs_model(
        self,
        rule_predictions: np.ndarray,
        model_predictions: np.ndarray,
    ) -> pd.DataFrame:
        """
        Builds a comparison table between rule-based and ML outputs.
        """

        return pd.DataFrame(
            {
                "rule_prediction": rule_predictions,
                "model_prediction": model_predictions,
                "agreement": (rule_predictions == model_predictions),
            }
        )


def add_temporal_columns(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Return a copy of the frame with `year` and `season` columns derived from timestamps."""

    if timestamp_column not in frame.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' is missing in frame.")

    enriched = frame.copy()
    enriched[timestamp_column] = pd.to_datetime(enriched[timestamp_column], errors="coerce")
    if enriched[timestamp_column].isna().any():
        raise ValueError(f"Timestamp column '{timestamp_column}' contains invalid values.")

    month = enriched[timestamp_column].dt.month
    enriched["year"] = enriched[timestamp_column].dt.year
    enriched["season"] = pd.Series(
        np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ["winter", "spring", "summer"],
            default="autumn",
        ),
        index=enriched.index,
    )
    return enriched


def _infer_expected_step(timestamps: pd.Series) -> pd.Timedelta:
    diffs = timestamps.diff().dropna()
    if diffs.empty:
        return pd.Timedelta(0)
    return cast(pd.Timedelta, diffs.median())


def _build_positive_spans(
    mask: np.ndarray, timestamps: pd.Series
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start_index: int | None = None
    previous_timestamp: pd.Timestamp | None = None
    expected_step = _infer_expected_step(timestamps)
    gap_threshold = expected_step * 1.5 if expected_step > pd.Timedelta(0) else pd.Timedelta(0)

    for index, (timestamp, is_positive) in enumerate(zip(timestamps, mask, strict=True)):
        if is_positive:
            if start_index is None:
                start_index = index
            elif (
                previous_timestamp is not None
                and gap_threshold > pd.Timedelta(0)
                and timestamp - previous_timestamp > gap_threshold
            ):
                spans.append((timestamps.iloc[start_index], previous_timestamp))
                start_index = index
        elif start_index is not None:
            spans.append((timestamps.iloc[start_index], previous_timestamp or timestamp))
            start_index = None
        previous_timestamp = timestamp

    if start_index is not None:
        spans.append(
            (timestamps.iloc[start_index], previous_timestamp or timestamps.iloc[start_index])
        )

    return spans


def _match_spans(
    true_spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    predicted_spans: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[int, int, int, list[float]]:
    matched_predicted: set[int] = set()
    onset_errors_hours: list[float] = []
    true_positive_events = 0

    for true_start, true_end in true_spans:
        match_index: int | None = None
        for predicted_index, (predicted_start, predicted_end) in enumerate(predicted_spans):
            if predicted_index in matched_predicted:
                continue
            if predicted_end < true_start or predicted_start > true_end:
                continue
            match_index = predicted_index
            break

        if match_index is None:
            continue

        matched_predicted.add(match_index)
        true_positive_events += 1
        predicted_start, _ = predicted_spans[match_index]
        onset_errors_hours.append(float((predicted_start - true_start).total_seconds() / 3600.0))

    false_negative_events = len(true_spans) - true_positive_events
    false_positive_events = len(predicted_spans) - len(matched_predicted)
    return true_positive_events, false_positive_events, false_negative_events, onset_errors_hours


def summarize_binary_event_predictions(
    frame: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    positive_label: str,
    timestamp_column: str = "timestamp",
    probability_column: str | None = None,
) -> dict[str, Any]:
    """Summarize row and event-level metrics for binary event detection outputs."""

    enriched = add_temporal_columns(frame, timestamp_column=timestamp_column)
    ordered = enriched.sort_values(timestamp_column).reset_index(drop=True)

    y_true = ordered[target_column].astype(str).eq(positive_label).to_numpy()
    y_pred = ordered[prediction_column].astype(str).eq(positive_label).to_numpy()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    true_spans = _build_positive_spans(y_true, ordered[timestamp_column])
    predicted_spans = _build_positive_spans(y_pred, ordered[timestamp_column])
    tp_events, fp_events, fn_events, onset_errors_hours = _match_spans(true_spans, predicted_spans)

    event_precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) else 0.0
    event_recall = tp_events / (tp_events + fn_events) if (tp_events + fn_events) else 0.0
    false_alarm_rate = fp_events / max(1, len(ordered))

    summary: dict[str, Any] = {
        "row_accuracy": float(accuracy),
        "row_precision": float(precision),
        "row_recall": float(recall),
        "row_f1": float(f1),
        "event_true_positives": tp_events,
        "event_false_positives": fp_events,
        "event_false_negatives": fn_events,
        "event_precision": float(event_precision),
        "event_recall": float(event_recall),
        "false_alarm_rate": float(false_alarm_rate),
        "mean_onset_error_hours": float(np.mean(onset_errors_hours))
        if onset_errors_hours
        else None,
        "median_onset_error_hours": float(np.median(onset_errors_hours))
        if onset_errors_hours
        else None,
        "true_event_count": len(true_spans),
        "predicted_event_count": len(predicted_spans),
    }

    if probability_column is not None and probability_column in ordered.columns:
        probabilities = ordered[probability_column].astype(float).to_numpy()
        summary["brier_score"] = float(brier_score_loss(y_true, probabilities))

    return summary


def summarize_by_period(
    frame: pd.DataFrame,
    *,
    period_column: str,
    target_column: str,
    prediction_column: str,
    positive_label: str,
    timestamp_column: str = "timestamp",
    probability_column: str | None = None,
) -> list[dict[str, Any]]:
    """Build summaries grouped by a temporal period column such as year or season."""

    enriched = add_temporal_columns(frame, timestamp_column=timestamp_column)
    rows: list[dict[str, Any]] = []
    for period_value, group in enriched.groupby(period_column, dropna=False):
        summary = summarize_binary_event_predictions(
            group,
            target_column=target_column,
            prediction_column=prediction_column,
            positive_label=positive_label,
            timestamp_column=timestamp_column,
            probability_column=probability_column,
        )
        summary[period_column] = period_value
        rows.append(summary)
    return rows
