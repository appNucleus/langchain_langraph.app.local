from __future__ import annotations

from fastapi import APIRouter, Request

from app import __version__
from app.settings import Settings

router = APIRouter()


@router.get("/")
async def root(request: Request) -> dict[str, str]:
    current_settings: Settings = request.app.state.settings
    return {
        "service": current_settings.app_name,
        "version": __version__,
        "status": "running",
        "liveness": "/health",
        "readiness": "/health/ready",
        "chat": "/api/chat",
        "stream": "/api/chat/stream",
        "inventory": "/api/inventory",
    }
