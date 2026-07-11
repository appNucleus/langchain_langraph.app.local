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
    events: list[SSEEvent] = []
    event_name: str | None = None
    event_id: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, event_id, data_lines
        if data_lines:
            events.append(SSEEvent(event=event_name, data="\n".join(data_lines), event_id=event_id))
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
        field, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "id":
            event_id = value
        elif field == "data":
            data_lines.append(value)
    flush()
    return events


def decode_json_response(text: str, content_type: str = "") -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise MCPProtocolError("MCP response body was empty.")

    candidates: list[str]
    if "text/event-stream" in content_type.lower() or stripped.startswith(("event:", "data:", ":")):
        candidates = [event.data for event in parse_sse(stripped) if event.data and event.data != "[DONE]"]
        if not candidates:
            raise MCPProtocolError("MCP SSE response contained no JSON data events.")
    else:
        candidates = [stripped]

    last_error: json.JSONDecodeError | None = None
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(payload, dict):
            return payload
    preview = candidates[-1][:300] if candidates else stripped[:300]
    raise MCPProtocolError(f"MCP response was not valid JSON: {preview}") from last_error
