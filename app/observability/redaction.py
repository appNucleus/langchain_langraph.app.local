from __future__ import annotations
from typing import Any
SENSITIVE={'authorization','x-api-key','api_key','confirmation_token','password','secret'}
def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k:('[REDACTED]' if k.lower() in SENSITIVE else redact(v)) for k,v in value.items()}
    if isinstance(value, list): return [redact(v) for v in value]
    return value
