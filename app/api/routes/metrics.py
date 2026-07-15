from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app import __version__
from app.api.dependencies import require_api_key
from app.observability.metrics import metrics
from app.settings import Settings

router = APIRouter(tags=["Status"])


@router.get("/api/metrics", summary="Get Application Metrics", dependencies=[Depends(require_api_key)])
async def application_metrics(request: Request) -> dict[str, Any]:
    current_settings: Settings = request.app.state.settings
    return {
        "service": current_settings.app_name,
        "version": __version__,
        **metrics.snapshot(),
    }
