"""Tests for continuous API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from src.api.main import app
from src.continuous.schemas import (
    HistoricalComparison,
    HorizonEvaluation,
    OnsetWindow,
    SourceStatus,
)


class _ServiceStub:
    def get_latest(self):
        from src.continuous.schemas import ContinuousEvaluationResult

        return ContinuousEvaluationResult(
            model_reference="configs/mlp_water_level.yaml",
            predicted_risk_level="high",
            risk_level_score=0.81,
            confidence_score=0.76,
            expected_onset_window=OnsetWindow(
                start_hours=1.0,
                end_hours=6.0,
                summary="Expected onset between +1h and +6h.",
            ),
            dominant_contributing_factor="flash_flood",
            historical_comparison=HistoricalComparison(
                event_type="flash_flood",
                historical_confidence=0.85,
                percentile_rank=0.72,
                summary="Historical confidence for flash_flood: 0.85.",
            ),
            source_status=[
                SourceStatus(name="open_meteo", ok=True, detail="ok"),
                SourceStatus(name="stormglass", ok=True, detail="ok"),
                SourceStatus(name="imgw", ok=True, detail="ok"),
            ],
            event_type="flash_flood",
            event_message="Possible flash flood detected.",
            summary="Risk=high, event=flash_flood.",
            generated_at=datetime.now(UTC),
            horizon_forecasts={
                "+1d": HorizonEvaluation(horizon="+1d", predicted_risk_level="high"),
                "+3d": HorizonEvaluation(horizon="+3d", predicted_risk_level="low"),
                "+7d": HorizonEvaluation(horizon="+7d"),
            },
            warnings=[],
        )

    def get_history(self, limit: int = 50):
        del limit
        return []

    def evaluate(self, persist: bool = True):
        del persist
        return self.get_latest()


def test_continuous_refresh_endpoint(monkeypatch) -> None:
    """Refresh endpoint should return continuous-evaluation payload."""

    from src.api import main as api_main

    monkeypatch.setattr(api_main, "build_service", lambda _config_path: _ServiceStub())
    client = TestClient(app)

    response = client.post(
        "/continuous/refresh",
        json={"config_path": "configs/mlp_water_level.yaml", "persist": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_risk_level"] == "high"
    assert payload["event_type"] == "flash_flood"


def test_continuous_forecast_endpoint(monkeypatch) -> None:
    """Forecast endpoint should return aggregated forecast."""

    from src.api import main as api_main

    monkeypatch.setattr(api_main, "build_service", lambda _config_path: _ServiceStub())
    client = TestClient(app)

    response = client.get("/continuous/forecast")
    assert response.status_code == 200
    payload = response.json()
    assert payload["current_risk_level"] == "high"
    assert "+1d" in payload["horizons"]
    assert "+3d" in payload["horizons"]
    assert "+7d" in payload["horizons"]


def test_continuous_status_endpoint(monkeypatch) -> None:
    """Status endpoint should return health info."""

    from src.api import main as api_main

    monkeypatch.setattr(api_main, "build_service", lambda _config_path: _ServiceStub())
    client = TestClient(app)

    response = client.get("/continuous/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["has_latest"] is True
    assert payload["current_risk_level"] == "high"
    assert len(payload["source_status"]) == 3


def test_continuous_prediction_by_horizon(monkeypatch) -> None:
    """Prediction-by-horizon endpoint should return detail for valid horizon."""

    from src.api import main as api_main

    monkeypatch.setattr(api_main, "build_service", lambda _config_path: _ServiceStub())
    client = TestClient(app)

    response = client.get("/continuous/predictions/+1d")
    assert response.status_code == 200
    payload = response.json()
    assert payload["horizon"] == "+1d"
    assert payload["available"] is True

    # Invalid horizon should 400
    response = client.get("/continuous/predictions/+2d")
    assert response.status_code == 400
