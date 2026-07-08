from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.logging_config import log_kv
from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPToolResult:
    tool: str
    ok: bool
    data: Any
    error: str | None = None


class MCPClientError(RuntimeError):
    pass


class MCPClient:
    """Async JSON-RPC client for the local MCP HTTP endpoint."""

    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport
        self._request_id = 0

    async def list_tools(self) -> list[dict[str, Any]]:
        data = await self._jsonrpc("tools/list", {})
        return list((data.get("result") or {}).get("tools") or [])

    async def health_check(self) -> MCPToolResult:
        return await self.call_tool("health_check", {})

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        try:
            data = await self._jsonrpc("tools/call", {"name": name, "arguments": arguments or {}})
            if "error" in data and data["error"]:
                return MCPToolResult(tool=name, ok=False, data=data, error=str(data["error"]))
            normalized = self._normalize_tool_result((data.get("result") or {}))
            return MCPToolResult(tool=name, ok=True, data=normalized)
        except Exception as exc:  # noqa: BLE001 - preserve tool failure as data for the LLM.
            error = _format_exception(exc)
            log_kv(logger, logging.ERROR, "mcp_client_tool_error", tool=name, error=error)
            return MCPToolResult(tool=name, ok=False, data=None, error=error)

    async def _jsonrpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self.settings.mcp_protocol_version,
        }
        url = self.settings.mcp_server_url.rstrip("/")
        log_kv(
            logger,
            logging.DEBUG,
            "mcp_jsonrpc_request",
            method=method,
            url=url,
            follow_redirects=self.settings.mcp_follow_redirects,
        )
        async with self._client() as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.history:
                log_kv(
                    logger,
                    logging.INFO,
                    "mcp_jsonrpc_redirect_followed",
                    method=method,
                    redirects=" -> ".join(str(item.url) for item in response.history + [response]),
                    final_url=str(response.url),
                    final_status=response.status_code,
                )
            log_kv(
                logger,
                logging.DEBUG,
                "mcp_jsonrpc_response",
                method=method,
                url=str(response.url),
                status_code=response.status_code,
                content_type=response.headers.get("content-type", ""),
            )
            response.raise_for_status()
            return self._decode_response(response)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.mcp_timeout_seconds),
            verify=self.settings.mcp_verify_tls,
            follow_redirects=self.settings.mcp_follow_redirects,
            transport=self._transport,
        )

    @staticmethod
    def _decode_response(response: httpx.Response) -> dict[str, Any]:
        text = response.text.strip()
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type or text.startswith("event:") or "\ndata:" in text:
            data_lines: list[str] = []
            for line in text.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").strip())
            if not data_lines:
                raise MCPClientError("MCP SSE response contained no data lines.")
            text = data_lines[-1]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise MCPClientError(f"MCP response was not valid JSON: {text[:300]}") from exc

    @staticmethod
    def _normalize_tool_result(result: dict[str, Any]) -> Any:
        """Normalize common MCP content formats into JSON-like Python data."""
        content = result.get("content")
        if isinstance(content, list):
            normalized_items: list[Any] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text") or "")
                    normalized_items.append(_maybe_json(text))
                else:
                    normalized_items.append(item)
            if len(normalized_items) == 1:
                return normalized_items[0]
            return normalized_items
        if "structuredContent" in result:
            return result["structuredContent"]
        return result


def _maybe_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped[0] in "[{\"":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return text
    return text


def _format_exception(exc: BaseException) -> str:
    text = str(exc).strip()
    if text:
        return f"{type(exc).__name__}: {text}"
    return f"{type(exc).__name__}: {exc!r}"
