"""API request and response schemas for continuous evaluation endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RefreshRequest(BaseModel):
    """Request payload for on-demand continuous refresh."""

    config_path: str = "configs/mlp_water_level.yaml"
    persist: bool = True


class HistoryQuery(BaseModel):
    """Utility schema for history query bounds."""

    limit: int = Field(default=50, ge=1, le=500)
