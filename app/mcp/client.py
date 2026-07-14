from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app import __version__
from app.logging_config import log_kv
from app.mcp.errors import MCPError, MCPHTTPStatusError, MCPProtocolError
from app.mcp.protocol import notification_envelope, request_envelope, validate_response
from app.mcp.result_parser import parse_tool_result
from app.mcp.session import MCPSessionManager
from app.mcp.transport import MCPHTTPTransport
from app.orchestration.execution_meter import (
    BudgetExceeded,
    get_current_execution_meter,
)
from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPToolResult:
    tool: str
    ok: bool
    data: Any
    error: str | None = None


class MCPClientError(MCPError):
    """Backward-compatible public MCP client exception."""


class MCPClient:
    """Session-aware async JSON-RPC client with physical-request metering."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        http_transport: MCPHTTPTransport | None = None,
    ) -> None:
        self.settings = settings
        self._transport = http_transport or MCPHTTPTransport(settings, transport=transport)
        self._request_id = 0
        self._id_lock = asyncio.Lock()
        self._session = MCPSessionManager(
            settings.mcp_protocol_version,
            settings.mcp_client_name,
            getattr(settings, "mcp_client_version", None) or __version__,
        )

    @property
    def session_id(self) -> str | None:
        return self._session.state.session_id

    async def start(self) -> None:
        await self._transport.start()
        if (
            self.settings.mcp_enabled
            and self.settings.mcp_session_enabled
            and self.settings.mcp_initialize_on_startup
        ):
            await self.initialize()

    async def aclose(self) -> None:
        await self._transport.close()
        self._session.reset()

    async def initialize(self) -> None:
        if not self.settings.mcp_session_enabled:
            return
        await self._session.ensure_initialized(self._send_for_session)

    async def list_tools(self) -> list[dict[str, Any]]:
        data = await self._rpc("tools/list", {})
        tools = (data or {}).get("tools") if isinstance(data, dict) else None
        if tools is None:
            return []
        if not isinstance(tools, list):
            raise MCPProtocolError("MCP tools/list result.tools must be a list.")
        return [item for item in tools if isinstance(item, dict)]

    async def health_check(self) -> MCPToolResult:
        return await self.call_tool("health_check", {})

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        try:
            result = await self._rpc(
                "tools/call", {"name": name, "arguments": arguments or {}}
            )
            parsed = parse_tool_result(result)
            return MCPToolResult(
                tool=name,
                ok=parsed.ok,
                data=parsed.data,
                error=parsed.error,
            )
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve dependency failure as tool data.
            error = _format_exception(exc)
            log_kv(
                logger,
                logging.ERROR,
                "mcp_client_tool_error",
                tool=name,
                error=error,
            )
            return MCPToolResult(tool=name, ok=False, data=None, error=error)

    async def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        if self.settings.mcp_session_enabled and method != "initialize":
            await self.initialize()
        try:
            result, _headers = await self._send_for_session(method, params, False)
            return result
        except MCPHTTPStatusError as exc:
            if (
                self.settings.mcp_session_enabled
                and self.session_id
                and exc.status_code in {404, 410}
            ):
                log_kv(
                    logger,
                    logging.WARNING,
                    "mcp_session_expired",
                    status_code=exc.status_code,
                )
                self._session.reset()
                await self.initialize()
                result, _headers = await self._send_for_session(method, params, False)
                return result
            raise

    async def _send_for_session(
        self,
        method: str,
        params: dict[str, Any],
        notification: bool,
    ) -> tuple[Any, dict[str, str]]:
        request_id: int | None = None
        if notification:
            payload = notification_envelope(method, params)
        else:
            request_id = await self._next_id()
            payload = request_envelope(request_id, method, params)
        headers = self._headers()
        log_kv(
            logger,
            logging.DEBUG,
            "mcp_jsonrpc_request",
            method=method,
            session=bool(self.session_id),
        )

        meter = get_current_execution_meter()
        if meter is not None:
            await meter.begin_tool_attempt()
        success = False
        timed_out = False
        try:
            if meter is None:
                body, response_headers = await self._transport.post(
                    payload,
                    headers=headers,
                    allow_empty=notification,
                )
            else:
                async with asyncio.timeout(max(0.001, meter.remaining_seconds())):
                    body, response_headers = await self._transport.post(
                        payload,
                        headers=headers,
                        allow_empty=notification,
                    )
            normalized_headers = {
                key.lower(): value for key, value in response_headers.items()
            }
            if notification:
                success = True
                return None, normalized_headers
            if body is None:
                raise MCPProtocolError(f"MCP method {method} returned an empty response.")
            response = validate_response(body, expected_id=request_id)
            if response.error is not None:
                raise MCPClientError(
                    f"MCP JSON-RPC error for {method}: {response.error}"
                )
            success = True
            return response.result, normalized_headers
        except asyncio.CancelledError:
            if meter is not None:
                meter.record_cancellation()
            raise
        except Exception as exc:
            timed_out = _looks_like_timeout(exc)
            raise
        finally:
            if meter is not None:
                await meter.finish_tool_attempt(
                    success=success,
                    timed_out=timed_out,
                )

    def _headers(self) -> dict[str, str]:
        protocol_version = (
            self._session.state.protocol_version or self.settings.mcp_protocol_version
        )
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": protocol_version,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    async def _next_id(self) -> int:
        async with self._id_lock:
            self._request_id += 1
            return self._request_id


def _looks_like_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return "timeout" in text or "timed out" in text


def _format_exception(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else f"{type(exc).__name__}: {exc!r}"
