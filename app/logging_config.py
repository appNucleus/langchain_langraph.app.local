from __future__ import annotations

import logging
import sys
from typing import Any

_APP_LOGGERS = [
    "app",
    "app.factory",
    "app.graph",
    "app.mcp.client",
    "app.mcp.transport",
    "app.mcp.session",
    "app.llm.ollama",
    "app.llm.resource_manager",
    "app.services.inventory",
]


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging so messages appear in docker logs/stdout.

    Uvicorn usually configures root logging before importing the app. This
    function is intentionally conservative: it adds a stdout handler only when
    no handler exists, then sets application logger levels explicitly.
    """

    normalized = (level or "INFO").upper()
    numeric_level = getattr(logging, normalized, logging.INFO)

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        root.addHandler(handler)
    root.setLevel(numeric_level)

    for name in _APP_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(numeric_level)
        logger.propagate = True


def log_kv(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Log compact key=value style events that are easy to grep in docker logs."""

    clean_fields = {key: _safe_value(value) for key, value in fields.items() if value is not None}
    suffix = " ".join(f"{key}={value}" for key, value in clean_fields.items())
    logger.log(level, "%s%s", event, f" {suffix}" if suffix else "")


def _safe_value(value: Any) -> str:
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) > 500:
        return text[:497] + "..."
    return text
