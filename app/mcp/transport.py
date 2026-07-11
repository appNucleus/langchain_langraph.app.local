from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.logging_config import log_kv
from app.mcp.errors import MCPHTTPStatusError, MCPTransportError
from app.mcp.sse import decode_json_response
from app.settings import Settings

import logging

logger = logging.getLogger(__name__)


class MCPHTTPTransport:
    """Long-lived HTTP transport for one MCP endpoint."""

    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self._custom_transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def start(self) -> None:
        await self._get_client()

    async def close(self) -> None:
        async with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def post(
        self,
        payload: dict[str, Any],
        *,
        headers: dict[str, str],
        allow_empty: bool = False,
    ) -> tuple[dict[str, Any] | None, httpx.Headers]:
        client = await self._get_client()
        url = self.settings.mcp_server_url.rstrip("/")
        try:
            response = await client.post(url, headers=headers, json=payload)
            if response.history:
                log_kv(
                    logger,
                    logging.INFO,
                    "mcp_redirect_followed",
                    redirects=" -> ".join(str(item.url) for item in [*response.history, response]),
                    final_status=response.status_code,
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise MCPHTTPStatusError(exc.response.status_code, f"{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise MCPTransportError(f"{type(exc).__name__}: {exc}") from exc

        if allow_empty and (response.status_code == 202 or not response.content.strip()):
            return None, response.headers
        return decode_json_response(response.text, response.headers.get("content-type", "")), response.headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                timeout = httpx.Timeout(
                    connect=self.settings.mcp_connect_timeout_seconds,
                    read=self.settings.mcp_read_timeout_seconds,
                    write=self.settings.mcp_write_timeout_seconds,
                    pool=self.settings.mcp_pool_timeout_seconds,
                )
                limits = httpx.Limits(
                    max_connections=self.settings.mcp_max_connections,
                    max_keepalive_connections=self.settings.mcp_max_keepalive_connections,
                    keepalive_expiry=self.settings.http_keepalive_expiry_seconds,
                )
                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    verify=self.settings.mcp_verify_tls,
                    follow_redirects=self.settings.mcp_follow_redirects,
                    transport=self._custom_transport,
                    limits=limits,
                )
        return self._client
