"""Evaluates rule-based event detectors and compares them against labels or model predictions."""

# pyright: reportArgumentType=false

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.events.schemas import EventDetection, EventEvaluation, EventType

DetectorFn = Callable[[pd.DataFrame], EventDetection]


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
