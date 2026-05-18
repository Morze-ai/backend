"""Tests for continuous inference service orchestration."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from src.continuous.clients import ClientResult
from src.continuous.schemas import ContinuousEvaluationResult
from src.continuous.service import ContinuousEvaluationService


class _OpenMeteoStub:
    def fetch_latest(self, latitude: float, longitude: float) -> ClientResult:
        del latitude, longitude
        return ClientResult(
            values={
                "temperature_c": 8.5,
                "humidity_percentage": 72.0,
                "rainfall_mm": 2.3,
                "pressure_hpa": 1009.0,
                "wind_speed": 6.2,
                "wind_direction": 210.0,
                "wind_u": -3.1,
                "wind_v": -5.3,
            },
            updated_at=datetime.now(UTC),
        )

    def fetch_forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        past_days: int = 10,
        forecast_days: int = 7,
    ) -> pd.DataFrame:
        del latitude, longitude, past_days, forecast_days
        return pd.DataFrame(columns=["timestamp"])


class _StormglassStub:
    def fetch_latest(self, latitude: float, longitude: float) -> ClientResult:
        del latitude, longitude
        return ClientResult(values={"water_level_m": 0.44}, updated_at=datetime.now(UTC))


class _ImgwStub:
    def fetch_latest(self) -> ClientResult:
        return ClientResult(
            values={
                "temperature_c": 9.0,
                "humidity_percentage": 70.0,
                "rainfall_mm": 1.7,
                "pressure_hpa": 1010.0,
            },
            updated_at=datetime.now(UTC),
        )


def test_continuous_service_evaluate_returns_required_fields() -> None:
    """Service should produce required output contract and persist artifacts."""

    service = ContinuousEvaluationService(
        config_path="configs/mlp_water_level.yaml",
        open_meteo_client=_OpenMeteoStub(),
        stormglass_client=_StormglassStub(),
        imgw_client=_ImgwStub(),
    )

    service.experiment.predict_one = lambda raw_values: ("high", [0.2, 0.8])

    # Stub the history loader to avoid FileNotFoundError when run in isolated CI environments
    feature_cols = service.config.data.feature_columns
    dummy_history = pd.DataFrame(
        {
            **{col: [0.0] for col in feature_cols},
            "timestamp": [pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)],
        }
    )
    service._load_history = lambda: dummy_history

    result = service.evaluate(persist=True)

    assert isinstance(result, ContinuousEvaluationResult)
    assert result.predicted_risk_level in {"low", "high"}
    assert 0.0 <= result.risk_level_score <= 1.0
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.expected_onset_window.start_hours >= 0.0
    assert result.expected_onset_window.end_hours >= result.expected_onset_window.start_hours
    assert result.dominant_contributing_factor
    assert result.historical_comparison.event_type
    assert len(result.source_status) >= 3

    latest = service.get_latest()
    assert latest is not None
    history = service.get_history(limit=5)
    assert len(history) >= 1
