from __future__ import annotations

from typing import Any

_SENSITIVE = {"authorization", "x-api-key", "api_key", "password", "token", "confirmation_token"}


def redact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: "[REDACTED]" if key.lower() in _SENSITIVE else item
        for key, item in value.items()
    }
