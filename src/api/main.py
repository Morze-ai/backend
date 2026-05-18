"""FastAPI application exposing continuous flood-risk endpoints."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src.api.schemas import (
    ForecastResponse,
    HorizonDetailResponse,
    RefreshRequest,
    StatusResponse,
)
from src.continuous.service import build_service

app = FastAPI(title="seaData Continuous Evaluation API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Health endpoint."""

    return {"status": "ok"}


@app.get("/continuous/latest")
def continuous_latest(
    config_path: str = Query(default="configs/mlp_water_level.yaml"),
) -> dict[str, object]:
    """Return latest persisted continuous result."""

    service = build_service(config_path)
    latest = service.get_latest()
    if latest is None:
        raise HTTPException(status_code=404, detail="No continuous result available yet.")
    return latest.model_dump(mode="json")


@app.get("/continuous/history")
def continuous_history(
    config_path: str = Query(default="configs/mlp_water_level.yaml"),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, object]]:
    """Return recent continuous history entries."""

    service = build_service(config_path)
    return [entry.model_dump(mode="json") for entry in service.get_history(limit=limit)]


@app.post("/continuous/refresh")
def continuous_refresh(request: RefreshRequest) -> dict[str, object]:
    """Trigger one on-demand continuous evaluation run."""

    service = build_service(request.config_path)
    result = service.evaluate(persist=request.persist)
    return result.model_dump(mode="json")


@app.get("/continuous/forecast", response_model=ForecastResponse)
def continuous_forecast(
    config_path: str = Query(default="configs/mlp_water_level.yaml"),
) -> ForecastResponse:
    """Return aggregated forecast with current + horizon predictions.

    Designed for frontend dashboard consumption. Returns the current risk
    assessment along with +1d, +3d, +7d horizon forecasts from the latest
    persisted evaluation.
    """

    service = build_service(config_path)
    latest = service.get_latest()
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail="No evaluation available. Run continuous-eval or POST /continuous/refresh first.",
        )

    return ForecastResponse(
        model_reference=latest.model_reference,
        generated_at=latest.generated_at,
        current_risk_level=latest.predicted_risk_level,
        current_risk_score=latest.risk_level_score,
        current_confidence=latest.confidence_score,
        current_event_type=latest.event_type,
        current_onset_window=latest.expected_onset_window,
        current_dominant_factor=latest.dominant_contributing_factor,
        current_historical_comparison=latest.historical_comparison,
        horizons=latest.horizon_forecasts,
        warnings=latest.warnings,
    )


@app.get("/continuous/status", response_model=StatusResponse)
def continuous_status(
    config_path: str = Query(default="configs/mlp_water_level.yaml"),
) -> StatusResponse:
    """Return system health: source freshness, last evaluation time, current risk."""

    service = build_service(config_path)
    latest = service.get_latest()

    if latest is None:
        return StatusResponse(has_latest=False)

    return StatusResponse(
        has_latest=True,
        last_evaluated_at=latest.generated_at,
        model_reference=latest.model_reference,
        source_status=latest.source_status,
        current_risk_level=latest.predicted_risk_level,
        warning_count=len(latest.warnings),
    )


@app.get("/continuous/predictions/{horizon}", response_model=HorizonDetailResponse)
def continuous_prediction_by_horizon(
    horizon: str,
    config_path: str = Query(default="configs/mlp_water_level.yaml"),
) -> HorizonDetailResponse:
    """Return prediction detail for a specific horizon (+1d, +3d, +7d)."""

    valid_horizons = {"+1d", "+3d", "+7d"}
    if horizon not in valid_horizons:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon '{horizon}'. Must be one of: {', '.join(sorted(valid_horizons))}",
        )

    service = build_service(config_path)
    latest = service.get_latest()
    if latest is None:
        raise HTTPException(status_code=404, detail="No evaluation available yet.")

    evaluation = latest.horizon_forecasts.get(horizon)
    if evaluation is None:
        return HorizonDetailResponse(
            horizon=horizon,
            available=False,
            message=f"No forecast data available for horizon {horizon}.",
        )

    return HorizonDetailResponse(
        horizon=horizon,
        available=True,
        evaluation=evaluation,
    )
