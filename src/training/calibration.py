"""Implements probability calibration techniques such as Platt Scaling (Logistic Calibration)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


class CalibrationManager:
    """Manages probability calibration for model outputs."""

    def __init__(self, method: str = "platt") -> None:
        self.method = method
        self.scaler = LogisticRegression(C=1e6, solver="lbfgs")  # type: ignore
        self._is_fitted = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> CalibrationManager:
        """
        Fits the calibration model using validation probabilities and ground truth.
        probs: array of shape (n_samples,) or (n_samples, 2)
        y_true: array of shape (n_samples,) with binary labels
        """
        if probs.ndim == 2:
            # For binary classification, take the positive class probability
            probs = probs[:, 1]

        # Reshape for sklearn
        X = probs.reshape(-1, 1)

        # Logistic Regression for Platt Scaling
        self.scaler.fit(X, y_true)
        self._is_fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Calibrates input probabilities.
        Returns calibrated probabilities of the same shape.
        """
        if not self._is_fitted:
            return probs

        original_shape = probs.shape
        if probs.ndim == 2:
            # Calibrate positive class
            p_pos = probs[:, 1].reshape(-1, 1)
            calibrated_p_pos = self.scaler.predict_proba(p_pos)[:, 1]

            calibrated_probs = np.zeros(original_shape)
            calibrated_probs[:, 1] = calibrated_p_pos
            calibrated_probs[:, 0] = 1.0 - calibrated_p_pos
            return calibrated_probs
        else:
            X = probs.reshape(-1, 1)
            return self.scaler.predict_proba(X)[:, 1]

    def evaluate_calibration(self, probs: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Computes Brier score to evaluate calibration quality."""
        if probs.ndim == 2:
            probs = probs[:, 1]

        brier = brier_score_loss(y_true, probs)
        return {"brier_score": float(brier)}
