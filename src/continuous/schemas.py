"""Typed schemas for continuous flood-risk inference outputs."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class SourceStatus(BaseModel):
    """Status and freshness metadata for one external data source."""

    name: str
    ok: bool
    detail: str = ""
    updated_at: datetime | None = None


class OnsetWindow(BaseModel):
    """Expected event onset window in hours ahead."""

    start_hours: float = Field(ge=0)
    end_hours: float = Field(ge=0)
    summary: str


class HistoricalComparison(BaseModel):
    """Contextual comparison to historical rule behavior."""

    event_type: str
    historical_confidence: float = Field(ge=0, le=1)
    percentile_rank: float = Field(ge=0, le=1)
    summary: str


class ContinuousEvaluationResult(BaseModel):
    """Unified output contract for scheduled and on-demand evaluation."""

    model_reference: str
    predicted_risk_level: str
    risk_level_score: float = Field(ge=0, le=1)
    confidence_score: float = Field(ge=0, le=1)
    expected_onset_window: OnsetWindow
    dominant_contributing_factor: str
    historical_comparison: HistoricalComparison
    source_status: list[SourceStatus]
    event_type: str
    event_message: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    summary: str
    horizon_forecasts: dict[str, HorizonEvaluation] = Field(default_factory=dict)
    report: ReportArtifacts | None = None
    warnings: list[str] = Field(default_factory=list)


class HorizonEvaluation(BaseModel):
    """Forecasted risk summary for a single horizon."""

    horizon: str
    target_timestamp: datetime | None = None
    predicted_risk_level: str | None = None
    forecasted_temperature_c: float | None = None
    forecasted_rainfall_mm: float | None = None
    forecasted_pressure_hpa: float | None = None
    forecasted_wind_speed_ms: float | None = None
    risk_level_score: float | None = Field(default=None, ge=0, le=1)
    confidence_score: float | None = Field(default=None, ge=0, le=1)
    expected_onset_window: OnsetWindow | None = None
    dominant_contributing_factor: str | None = None
    main_cause: str | None = None
    historical_comparison: HistoricalComparison | None = None
    event_type: str | None = None
    event_message: str | None = None
    top_features: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReportArtifacts(BaseModel):
    """References to generated report artifacts."""

    markdown_path: str | None = None
    plot_path: str | None = None
