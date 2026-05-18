"""API request and response schemas for continuous evaluation endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.continuous.schemas import (
    HistoricalComparison,
    HorizonEvaluation,
    OnsetWindow,
    SourceStatus,
)


class RefreshRequest(BaseModel):
    """Request payload for on-demand continuous refresh."""

    config_path: str = "configs/mlp_water_level.yaml"
    persist: bool = True


class HistoryQuery(BaseModel):
    """Utility schema for history query bounds."""

    limit: int = Field(default=50, ge=1, le=500)


class ForecastResponse(BaseModel):
    """Aggregated forecast response for all horizons."""

    model_reference: str
    generated_at: datetime
    current_risk_level: str
    current_risk_score: float = Field(ge=0, le=1)
    current_confidence: float = Field(ge=0, le=1)
    current_event_type: str
    current_onset_window: OnsetWindow
    current_dominant_factor: str
    current_historical_comparison: HistoricalComparison
    horizons: dict[str, HorizonEvaluation]
    warnings: list[str] = Field(default_factory=list)


class StatusResponse(BaseModel):
    """Health and freshness status for the continuous evaluation system."""

    has_latest: bool
    last_evaluated_at: datetime | None = None
    model_reference: str | None = None
    source_status: list[SourceStatus] = Field(default_factory=list)
    current_risk_level: str | None = None
    warning_count: int = 0


class HorizonDetailResponse(BaseModel):
    """Detailed prediction for a single forecast horizon."""

    horizon: str
    available: bool
    evaluation: HorizonEvaluation | None = None
    message: str = ""
