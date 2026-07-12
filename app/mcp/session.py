from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.mcp.errors import MCPSessionError

RPCSender = Callable[
    [str, dict[str, Any], bool],
    Awaitable[tuple[Any, dict[str, str]]],
]

# The repository already interoperates with both protocol generations. A client
# must reject arbitrary server-selected versions, but it can support more than
# the version it initially offers.
SUPPORTED_PROTOCOL_VERSIONS = frozenset({"2025-06-18", "2025-03-26", "2024-11-05"})


@dataclass
class MCPSessionState:
    initialized: bool = False
    session_id: str | None = None
    protocol_version: str | None = None
    server_info: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None


class MCPSessionManager:
    """Single-flight MCP initialization and session-expiration recovery."""

    def __init__(
        self,
        protocol_version: str,
        client_name: str,
        client_version: str,
        *,
        supported_versions: frozenset[str] | None = None,
    ) -> None:
        self.requested_protocol_version = protocol_version
        self.client_name = client_name
        self.client_version = client_version
        self.supported_versions = (
            supported_versions or SUPPORTED_PROTOCOL_VERSIONS | {protocol_version}
        )
        self.state = MCPSessionState()
        self._lock = asyncio.Lock()
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    async def ensure_initialized(self, sender: RPCSender) -> MCPSessionState:
        if self.state.initialized:
            return self.state
        async with self._lock:
            if self.state.initialized:
                return self.state
            return await self._initialize_locked(sender)

    async def recover_expired(
        self,
        sender: RPCSender,
        *,
        observed_generation: int,
    ) -> MCPSessionState:
        """Reinitialize once when the observed session is still current.

        Concurrent callers that saw the same expired session wait on the lock.
        Only the first caller resets and initializes; later callers reuse the new
        session rather than starting additional initialize handshakes.
        """

        async with self._lock:
            if self.state.initialized and self._generation != observed_generation:
                return self.state
            self._reset_unlocked()
            return await self._initialize_locked(sender)

    async def _initialize_locked(self, sender: RPCSender) -> MCPSessionState:
        try:
            result, headers = await sender(
                "initialize",
                {
                    "protocolVersion": self.requested_protocol_version,
                    "capabilities": {},
                    "clientInfo": {
                        "name": self.client_name,
                        "version": self.client_version,
                    },
                },
                False,
            )
            if not isinstance(result, dict):
                raise MCPSessionError("MCP initialize returned an invalid result.")

            negotiated = result.get("protocolVersion")
            if not isinstance(negotiated, str) or not negotiated:
                raise MCPSessionError(
                    "MCP initialize result is missing protocolVersion."
                )
            if negotiated not in self.supported_versions:
                raise MCPSessionError(
                    f"MCP server selected unsupported protocol version {negotiated!r}."
                )

            session_id = headers.get("mcp-session-id")
            self.state = MCPSessionState(
                initialized=False,
                session_id=session_id or None,
                protocol_version=negotiated,
                server_info=(
                    result.get("serverInfo")
                    if isinstance(result.get("serverInfo"), dict)
                    else None
                ),
                capabilities=(
                    result.get("capabilities")
                    if isinstance(result.get("capabilities"), dict)
                    else None
                ),
            )

            # The initialized notification must carry the negotiated version and,
            # when issued by the server, the session header.
            await sender("notifications/initialized", {}, True)
            self.state.initialized = True
            self._generation += 1
            return self.state
        except Exception:
            self._reset_unlocked()
            raise

    def reset(self) -> None:
        self._reset_unlocked()

    def _reset_unlocked(self) -> None:
        self.state = MCPSessionState()
        self._generation += 1
