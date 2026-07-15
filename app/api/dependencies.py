from __future__ import annotations

from typing import Annotated

from fastapi import Header, HTTPException, Request, status

from app.settings import Settings


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Enforce the configured API key on protected routes."""

    current_settings: Settings = request.app.state.settings
    if current_settings.api_key and x_api_key != current_settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
