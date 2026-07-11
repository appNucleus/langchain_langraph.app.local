from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.mcp.errors import MCPSessionError


RPCSender = Callable[[str, dict[str, Any], bool], Awaitable[tuple[Any, dict[str, str]]]]


@dataclass
class MCPSessionState:
    initialized: bool = False
    session_id: str | None = None
    protocol_version: str | None = None
    server_info: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None


class MCPSessionManager:
    def __init__(self, protocol_version: str, client_name: str, client_version: str) -> None:
        self.requested_protocol_version = protocol_version
        self.client_name = client_name
        self.client_version = client_version
        self.state = MCPSessionState()
        self._lock = asyncio.Lock()

    async def ensure_initialized(self, sender: RPCSender) -> MCPSessionState:
        if self.state.initialized:
            return self.state
        async with self._lock:
            if self.state.initialized:
                return self.state
            try:
                result, headers = await sender(
                    "initialize",
                    {
                        "protocolVersion": self.requested_protocol_version,
                        "capabilities": {},
                        "clientInfo": {"name": self.client_name, "version": self.client_version},
                    },
                    False,
                )
                if not isinstance(result, dict):
                    raise MCPSessionError("MCP initialize returned an invalid result.")
                self.state.session_id = headers.get("mcp-session-id")
                self.state.protocol_version = str(result.get("protocolVersion") or self.requested_protocol_version)
                self.state.server_info = result.get("serverInfo") if isinstance(result.get("serverInfo"), dict) else None
                self.state.capabilities = result.get("capabilities") if isinstance(result.get("capabilities"), dict) else None
                await sender("notifications/initialized", {}, True)
                self.state.initialized = True
                return self.state
            except Exception:
                self.reset()
                raise

    def reset(self) -> None:
        self.state = MCPSessionState()
