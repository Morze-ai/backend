"""Verifies temporal and event-level evaluation helpers."""

import numpy as np
import pandas as pd

from src.events.evaluator import (
    add_temporal_columns,
    summarize_binary_event_predictions,
    summarize_by_period,
)


def test_add_temporal_columns_derives_year_and_season() -> None:
    """Test that timestamps are expanded into year and season labels."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]),
            "value": [1, 2, 3, 4],
        }
    )

    enriched = add_temporal_columns(frame)

    assert enriched["year"].tolist() == [2024, 2024, 2024, 2024]
    assert enriched["season"].tolist() == ["winter", "spring", "summer", "autumn"]


def test_summarize_binary_event_predictions_matches_episodes() -> None:
    """Test episode matching, onset error, false alarms, and calibration summary."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "target": ["low", "high", "high", "low", "low", "high", "high", "low", "low", "low"],
            "predicted_class": [
                "low",
                "high",
                "high",
                "low",
                "high",
                "high",
                "high",
                "low",
                "high",
                "high",
            ],
            "prob_high": [0.05, 0.92, 0.88, 0.12, 0.61, 0.86, 0.81, 0.08, 0.77, 0.79],
        }
    )

    summary = summarize_binary_event_predictions(
        frame,
        target_column="target",
        prediction_column="predicted_class",
        positive_label="high",
        timestamp_column="timestamp",
        probability_column="prob_high",
    )

    assert summary["true_event_count"] == 2
    assert summary["predicted_event_count"] == 3
    assert summary["event_true_positives"] == 2
    assert summary["event_false_positives"] == 1
    assert summary["event_false_negatives"] == 0
    assert summary["event_recall"] == 1.0
    assert summary["event_precision"] == 2 / 3
    assert summary["false_alarm_rate"] == 0.1
    assert summary["mean_onset_error_hours"] == -12.0
    assert "brier_score" in summary


def test_summarize_by_period_returns_seasonal_breakdowns() -> None:
    """Test that period summaries can be built from the same event-level helper."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "target": ["low", "high", "high", "low", "low", "low"],
            "predicted_class": ["low", "high", "high", "low", "low", "low"],
            "prob_high": np.linspace(0.1, 0.9, 6),
        }
    )

    by_year = summarize_by_period(
        frame,
        period_column="year",
        target_column="target",
        prediction_column="predicted_class",
        positive_label="high",
        timestamp_column="timestamp",
        probability_column="prob_high",
    )

    by_season = summarize_by_period(
        frame,
        period_column="season",
        target_column="target",
        prediction_column="predicted_class",
        positive_label="high",
        timestamp_column="timestamp",
        probability_column="prob_high",
    )

    assert by_year[0]["year"] == 2024
    assert any(row["season"] == "winter" for row in by_season)
