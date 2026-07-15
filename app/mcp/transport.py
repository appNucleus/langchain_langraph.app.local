from __future__ import annotations

import asyncio
import logging
from time import monotonic
from typing import Any

import httpx2 as httpx

from app.logging_config import log_kv
from app.mcp.errors import MCPHTTPStatusError, MCPProtocolError, MCPTransportError
from app.mcp.sse import decode_json_messages
from app.observability.metrics import metrics
from app.settings import Settings

logger = logging.getLogger(__name__)


class MCPHTTPTransport:
    """Long-lived, concurrency-bounded HTTP transport for one MCP endpoint."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._custom_transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._concurrency = asyncio.Semaphore(max(1, settings.mcp_max_concurrency))

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
    ) -> tuple[list[dict[str, Any]] | None, httpx.Headers]:
        client = await self._get_client()
        url = self.settings.mcp_server_url.rstrip("/")
        started = monotonic()
        metrics.inc("mcp.http_post_requests")

        async with self._concurrency:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.history:
                    log_kv(
                        logger,
                        logging.INFO,
                        "mcp_redirect_followed",
                        redirects=" -> ".join(
                            str(item.url) for item in [*response.history, response]
                        ),
                        final_status=response.status_code,
                    )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                metrics.inc("mcp.http_status_error")
                body = exc.response.text.strip().replace("\n", " ")[:500]
                suffix = f"; response={body}" if body else ""
                raise MCPHTTPStatusError(
                    exc.response.status_code,
                    f"{type(exc).__name__}: {exc}{suffix}",
                ) from exc
            except httpx.TimeoutException as exc:
                metrics.inc("mcp.http_timeout")
                raise MCPTransportError(
                    f"MCP HTTP timeout ({type(exc).__name__}): {exc}"
                ) from exc
            except httpx.HTTPError as exc:
                metrics.inc("mcp.http_transport_error")
                raise MCPTransportError(f"{type(exc).__name__}: {exc}") from exc
            finally:
                metrics.observe("mcp.http_post_seconds", monotonic() - started)

        if allow_empty and (response.status_code == 202 or not response.content.strip()):
            return None, response.headers
        if not response.content.strip():
            raise MCPProtocolError("MCP request returned an empty response body.")

        messages = decode_json_messages(
            response.text,
            response.headers.get("content-type", ""),
        )
        return messages, response.headers

    async def delete(self, *, headers: dict[str, str]) -> bool:
        """Best-effort MCP session termination.

        Servers may return 405 when explicit session deletion is unsupported; that
        is not treated as a shutdown failure.
        """

        client = await self._get_client()
        url = self.settings.mcp_server_url.rstrip("/")
        started = monotonic()
        metrics.inc("mcp.http_delete_requests")
        async with self._concurrency:
            try:
                response = await client.request("DELETE", url, headers=headers)
            except httpx.HTTPError as exc:
                metrics.inc("mcp.http_delete_error")
                log_kv(
                    logger,
                    logging.WARNING,
                    "mcp_session_delete_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                return False
            finally:
                metrics.observe("mcp.http_delete_seconds", monotonic() - started)

        if response.status_code in {200, 202, 204, 404, 405}:
            return response.status_code not in {404, 405}

        log_kv(
            logger,
            logging.WARNING,
            "mcp_session_delete_rejected",
            status_code=response.status_code,
        )
        return False

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
                    max_keepalive_connections=(
                        self.settings.mcp_max_keepalive_connections
                    ),
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
