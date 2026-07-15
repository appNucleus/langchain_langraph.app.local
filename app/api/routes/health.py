from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app import __version__
from app.api.exception_handlers import safe_error
from app.graph import ChatAgent
from app.services.inventory import InventoryService
from app.settings import Settings

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    current_settings: Settings = request.app.state.settings
    return {
        "status": "ok",
        "service": current_settings.app_name,
        "version": __version__,
    }


@router.get("/health/live")
async def live_health(request: Request) -> JSONResponse:
    current_settings: Settings = request.app.state.settings
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "alive",
            "service": current_settings.app_name,
            "version": __version__,
        },
    )


@router.get("/health/ready")
async def ready_health(request: Request) -> JSONResponse:
    current_agent: ChatAgent = request.app.state.chat_agent
    current_settings: Settings = request.app.state.settings
    service: InventoryService = request.app.state.inventory_service
    payload: dict[str, Any] = {
        "status": "ready",
        "service": current_settings.app_name,
        "version": __version__,
        "dependencies": {},
    }
    ready = True
    live_inventory = await service.load()

    if current_settings.llm_backend == "ollama":
        error = live_inventory.errors.get("ollama")
        available = not error and bool(live_inventory.model_names)
        payload["dependencies"]["ollama"] = {
            "status": "available" if available else "unavailable",
            "model_count": len(live_inventory.model_names),
            "cached": live_inventory.cached,
            "error": (
                error
                if current_settings.expose_internal_health_details
                else ("Dependency unavailable." if error else None)
            ),
        }
        if current_settings.ollama_required:
            ready = ready and available

    if current_settings.mcp_enabled:
        error = live_inventory.errors.get("mcp")
        available = not error
        payload["dependencies"]["mcp"] = {
            "status": "available" if available else "unavailable",
            "tool_count": len(live_inventory.tool_names),
            "cached": live_inventory.cached,
            "error": (
                error
                if current_settings.expose_internal_health_details
                else ("Dependency unavailable." if error else None)
            ),
        }
        if current_settings.mcp_required:
            ready = ready and available

    try:
        persistence = await current_agent.persistence_health()
        payload["dependencies"]["persistence"] = persistence
        conversation_ok = (persistence.get("conversation") or {}).get(
            "status"
        ) == "available"
        runs_ok = (persistence.get("runs") or {}).get("status") == "available"
        checkpoint_ok = (persistence.get("checkpoint") or {}).get(
            "status"
        ) == "available"
        artifacts_ok = (persistence.get("artifacts") or {}).get("status") in {
            "available",
            "disabled",
        }
        if current_settings.persistence_required:
            ready = ready and conversation_ok and runs_ok and checkpoint_ok
        if current_settings.artifact_storage_required:
            ready = ready and artifacts_ok
    except Exception as exc:
        if (
            current_settings.persistence_required
            or current_settings.artifact_storage_required
        ):
            ready = False
        payload["dependencies"]["persistence"] = {
            "status": "unavailable",
            "error": safe_error(
                exc,
                expose=current_settings.expose_internal_health_details,
            ),
        }

    startup_status = getattr(current_agent, "dependency_startup_status", None)
    if callable(startup_status):
        payload["startup"] = startup_status()
    if not ready:
        payload["status"] = "not_ready"
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content=payload,
    )
