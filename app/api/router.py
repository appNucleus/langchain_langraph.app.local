from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import chat, health, inventory, metrics, root

router = APIRouter()
router.include_router(root.router)
router.include_router(health.router)
router.include_router(inventory.router)
router.include_router(metrics.router)
router.include_router(chat.router)
