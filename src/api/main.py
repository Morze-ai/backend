"""FastAPI application exposing continuous flood-risk endpoints."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src.api.schemas import RefreshRequest
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
