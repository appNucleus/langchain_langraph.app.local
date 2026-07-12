from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.mcp.errors import MCPProtocolError


@dataclass(frozen=True)
class SSEEvent:
    event: str | None
    data: str
    event_id: str | None = None


def parse_sse(text: str) -> list[SSEEvent]:
    """Parse an SSE payload, preserving multiline ``data:`` fields."""

    events: list[SSEEvent] = []
    event_name: str | None = None
    event_id: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, event_id, data_lines
        if data_lines:
            events.append(
                SSEEvent(
                    event=event_name,
                    data="\n".join(data_lines),
                    event_id=event_id,
                )
            )
        event_name = None
        event_id = None
        data_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            flush()
            continue
        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]

        if field == "event":
            event_name = value
        elif field == "id":
            # NUL is not valid in an SSE id; ignore such values.
            if "\x00" not in value:
                event_id = value
        elif field == "data":
            data_lines.append(value)

    flush()
    return events


def decode_json_messages(text: str, content_type: str = "") -> list[dict[str, Any]]:
    """Decode all JSON-RPC objects from JSON or SSE response content."""

    stripped = text.strip()
    if not stripped:
        raise MCPProtocolError("MCP response body was empty.")

    is_sse = (
        "text/event-stream" in content_type.lower()
        or stripped.startswith(("event:", "data:", ":"))
    )
    candidates = (
        [event.data for event in parse_sse(stripped) if event.data != "[DONE]"]
        if is_sse
        else [stripped]
    )
    if not candidates:
        raise MCPProtocolError("MCP SSE response contained no JSON data events.")

    messages: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            preview = candidate[:300]
            raise MCPProtocolError(
                f"MCP response contained invalid JSON: {preview}"
            ) from exc

        values = payload if isinstance(payload, list) else [payload]
        for value in values:
            if not isinstance(value, dict):
                raise MCPProtocolError(
                    "MCP JSON response items must be JSON objects."
                )
            messages.append(value)

    if not messages:
        raise MCPProtocolError("MCP response contained no JSON-RPC objects.")
    return messages


def decode_json_response(text: str, content_type: str = "") -> dict[str, Any]:
    """Backward-compatible helper returning the final decoded JSON object."""

    return decode_json_messages(text, content_type)[-1]
