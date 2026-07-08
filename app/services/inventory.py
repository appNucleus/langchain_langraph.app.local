from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.logging_config import log_kv
from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeInventory:
    """Live view of the resources the app can use right now."""

    models: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def model_names(self) -> list[str]:
        names: list[str] = []
        for item in self.models:
            name = item.get("name") or item.get("model")
            if isinstance(name, str) and name and name not in names:
                names.append(name)
        return names

    @property
    def tool_names(self) -> list[str]:
        names: list[str] = []
        for item in self.tools:
            name = item.get("name")
            if isinstance(name, str) and name and name not in names:
                names.append(name)
        return names

    def has_tool(self, name: str) -> bool:
        return name in set(self.tool_names)

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "model_names": self.model_names,
            "tools": self.tools,
            "tool_names": self.tool_names,
            "errors": self.errors,
        }


class InventoryService:
    """Loads live Ollama model and MCP tool inventory.

    This service deliberately makes best-effort calls. The chat endpoint should
    still answer when one dependency is temporarily down, but the response
    metadata will expose the dependency error.
    """

    def __init__(self, settings: Settings, ollama_client: Any, mcp_client: Any) -> None:
        self.settings = settings
        self.ollama = ollama_client
        self.mcp = mcp_client

    async def load(self) -> RuntimeInventory:
        models: list[dict[str, Any]] = []
        tools: list[dict[str, Any]] = []
        errors: dict[str, str] = {}

        if self.settings.llm_backend == "ollama":
            try:
                if hasattr(self.ollama, "list_models"):
                    models = await self.ollama.list_models()
                else:
                    health = await self.ollama.health()
                    models = [{"name": name} for name in health.get("models", [])]
            except Exception as exc:  # noqa: BLE001 - keep inventory best-effort.
                errors["ollama"] = _format_exception(exc)
                log_kv(logger, logging.WARNING, "inventory_ollama_error", error=errors["ollama"])

        if self.settings.mcp_enabled:
            try:
                tools = await self.mcp.list_tools()
            except Exception as exc:  # noqa: BLE001 - keep inventory best-effort.
                errors["mcp"] = _format_exception(exc)
                log_kv(logger, logging.WARNING, "inventory_mcp_error", error=errors["mcp"])

        return RuntimeInventory(models=models, tools=tools, errors=errors)


class ModelSelector:
    """Resolve model roles against the models that are actually live in Ollama."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def resolve(self, key: str | None, inventory: RuntimeInventory | None = None) -> str:
        key = key or "general"
        configured = self.settings.model_for_key(key)
        available = inventory.model_names if inventory else []
        if not available or configured in available:
            return configured

        for candidate in self._preferred_candidates(key):
            if candidate in available:
                return candidate

        return available[0] if available else self.settings.model_fallback

    def configured_roles(self, inventory: RuntimeInventory | None = None) -> dict[str, dict[str, Any]]:
        roles = ["planner", "simple", "general", "search", "reasoning", "fast_reasoning", "heavy", "synthesis", "writer", "classifier", "vision", "fallback"]
        available = set(inventory.model_names if inventory else [])
        output: dict[str, dict[str, Any]] = {}
        for role in roles:
            configured = self.settings.model_for_key(role)
            resolved = self.resolve(role, inventory)
            output[role] = {
                "configured": configured,
                "resolved": resolved,
                "available": not available or resolved in available,
            }
        output["embedding"] = {
            "configured": self.settings.embedding_model,
            "resolved": self.settings.embedding_model if not available or self.settings.embedding_model in available else None,
            "available": not available or self.settings.embedding_model in available,
            "chat_model": False,
        }
        return output

    def _preferred_candidates(self, key: str) -> list[str]:
        common = [
            self.settings.model_fallback,
            self.settings.model_general,
            self.settings.model_search,
            self.settings.model_reasoning,
            self.settings.model_fast_reasoning,
            self.settings.model_simple,
        ]
        preferences = {
            "planner": [self.settings.model_planner, self.settings.model_general, self.settings.model_classifier, self.settings.model_search],
            "simple": [self.settings.model_simple, self.settings.model_general, self.settings.model_fallback],
            "general": [self.settings.model_general, self.settings.model_simple, self.settings.model_search, self.settings.model_fallback],
            "search": [self.settings.model_search, self.settings.model_general, self.settings.model_fallback],
            "reasoning": [self.settings.model_reasoning, self.settings.model_fast_reasoning, self.settings.model_heavy, self.settings.model_search],
            "fast_reasoning": [self.settings.model_fast_reasoning, self.settings.model_reasoning, self.settings.model_general, self.settings.model_fallback],
            "heavy": [self.settings.model_heavy, self.settings.model_synthesis, self.settings.model_reasoning, self.settings.model_search],
            "synthesis": [self.settings.model_synthesis, self.settings.model_heavy, self.settings.model_reasoning, self.settings.model_search],
            "writer": [self.settings.model_writer, self.settings.model_general, self.settings.model_synthesis, self.settings.model_fallback],
            "classifier": [self.settings.model_classifier, self.settings.model_general, self.settings.model_simple, self.settings.model_fallback],
            "vision": [self.settings.model_vision, self.settings.model_general, self.settings.model_fallback],
            "fallback": [self.settings.model_fallback, self.settings.model_general],
        }
        merged: list[str] = []
        for candidate in [*preferences.get(key, []), *common]:
            if candidate and candidate not in merged:
                merged.append(candidate)
        return merged


def build_inventory_payload(settings: Settings, inventory: RuntimeInventory, selector: ModelSelector) -> dict[str, Any]:
    return {
        "service": {
            "name": settings.app_name,
            "environment": settings.environment,
            "backend": settings.llm_backend,
        },
        "ollama": {
            "base_url": settings.ollama_base_url,
            "models": inventory.models,
            "model_names": inventory.model_names,
            "configured_roles": selector.configured_roles(inventory),
            "model_task_catalog": settings.model_role_catalog(),
        },
        "mcp": {
            "enabled": settings.mcp_enabled,
            "server_url": settings.mcp_server_url,
            "tools": inventory.tools,
            "tool_names": inventory.tool_names,
        },
        "errors": inventory.errors,
    }


def _format_exception(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else f"{type(exc).__name__}: {exc!r}"
