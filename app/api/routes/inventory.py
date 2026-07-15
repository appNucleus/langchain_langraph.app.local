from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import require_api_key
from app.graph import ChatAgent
from app.services.inventory import InventoryService, build_inventory_payload
from app.settings import Settings

router = APIRouter()


@router.get("/api/inventory", dependencies=[Depends(require_api_key)])
async def inventory(request: Request) -> dict[str, object]:
    current_agent: ChatAgent = request.app.state.chat_agent
    current_settings: Settings = request.app.state.settings
    service: InventoryService = request.app.state.inventory_service
    live_inventory = await service.load()
    return build_inventory_payload(
        current_settings,
        live_inventory,
        current_agent.selector,
    )
