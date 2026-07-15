from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import root, health, inventory, metrics, chat

router = APIRouter()
router.include_router(root.router)
router.include_router(health.router)
router.include_router(inventory.router)
router.include_router(metrics.router)
router.include_router(chat.router)
