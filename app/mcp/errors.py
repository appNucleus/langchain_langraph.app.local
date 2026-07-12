from __future__ import annotations


class MCPError(RuntimeError):
    """Base error for MCP transport, session, and protocol failures."""


class MCPTransportError(MCPError):
    pass


class MCPHTTPStatusError(MCPTransportError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class MCPProtocolError(MCPError):
    pass


class MCPSessionError(MCPError):
    pass


class MCPSessionExpiredError(MCPSessionError):
    """The server expired a session and the original call was not safely replayed."""


class MCPToolError(MCPError):
    pass


class MCPResponseMismatchError(MCPProtocolError):
    pass
