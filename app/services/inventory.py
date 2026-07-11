from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from app.llm.model_registry import capabilities_for
from app.logging_config import log_kv
from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeInventory:
    models: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    cached: bool = False
    age_seconds: float = 0.0

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
        return name in self.tool_names

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "model_names": self.model_names,
            "tools": self.tools,
            "tool_names": self.tool_names,
            "errors": self.errors,
            "cached": self.cached,
            "age_seconds": round(self.age_seconds, 3),
        }


class InventoryService:
    """Concurrent, short-lived cache for Ollama and MCP capability discovery."""

    def __init__(self, settings: Settings, ollama_client: Any, mcp_client: Any) -> None:
        self.settings = settings
        self.ollama = ollama_client
        self.mcp = mcp_client
        self._cache: RuntimeInventory | None = None
        self._cached_at = 0.0
        self._lock = asyncio.Lock()

    async def load(self, *, force_refresh: bool = False) -> RuntimeInventory:
        now = monotonic()
        if self._is_fresh(now, force_refresh):
            return self._with_cache_metadata(self._cache, cached=True)  # type: ignore[arg-type]

        async with self._lock:
            now = monotonic()
            if self._is_fresh(now, force_refresh):
                return self._with_cache_metadata(self._cache, cached=True)  # type: ignore[arg-type]

            models_result, tools_result = await asyncio.gather(
                self._load_models(),
                self._load_tools(),
            )
            models, model_error = models_result
            tools, tool_error = tools_result
            errors = {
                key: value
                for key, value in (("ollama", model_error), ("mcp", tool_error))
                if value
            }

            if (
                errors
                and self._cache is not None
                and now - self._cached_at <= self.settings.inventory_stale_if_error_seconds
            ):
                stale = RuntimeInventory(
                    models=self._cache.models,
                    tools=self._cache.tools,
                    errors=errors,
                    cached=True,
                    age_seconds=now - self._cached_at,
                )
                log_kv(logger, logging.WARNING, "inventory_stale_cache_used", errors=errors)
                return stale

            fresh = RuntimeInventory(models=models, tools=tools, errors=errors)
            self._cache = fresh
            self._cached_at = now
            return fresh

    def invalidate(self) -> None:
        self._cache = None
        self._cached_at = 0.0

    def _is_fresh(self, now: float, force_refresh: bool) -> bool:
        return (
            not force_refresh
            and self._cache is not None
            and now - self._cached_at <= self.settings.inventory_cache_ttl_seconds
        )

    async def _load_models(self) -> tuple[list[dict[str, Any]], str | None]:
        if self.settings.llm_backend != "ollama":
            return [], None
        try:
            return await self.ollama.list_models(), None
        except Exception as exc:  # noqa: BLE001
            error = _format_exception(exc)
            log_kv(logger, logging.WARNING, "inventory_ollama_error", error=error)
            return [], error

    async def _load_tools(self) -> tuple[list[dict[str, Any]], str | None]:
        if not self.settings.mcp_enabled:
            return [], None
        try:
            return await self.mcp.list_tools(), None
        except Exception as exc:  # noqa: BLE001
            error = _format_exception(exc)
            log_kv(logger, logging.WARNING, "inventory_mcp_error", error=error)
            return [], error

    def _with_cache_metadata(
        self,
        inventory: RuntimeInventory,
        *,
        cached: bool,
    ) -> RuntimeInventory:
        return RuntimeInventory(
            models=inventory.models,
            tools=inventory.tools,
            errors=inventory.errors,
            cached=cached,
            age_seconds=max(0.0, monotonic() - self._cached_at),
        )


class ModelSelector:
    """Resolve roles only to compatible live models."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def resolve(self, key: str | None, inventory: RuntimeInventory | None = None) -> str:
        role = key or "general"
        configured = self.settings.model_for_key(role)
        available = inventory.model_names if inventory else []
        if not available:
            return configured

        for candidate in self._preferred_candidates(role):
            if candidate in available and self._compatible(role, candidate):
                return candidate

        fallback = self.settings.model_fallback
        if fallback in available and self._compatible(role, fallback):
            return fallback
        raise RuntimeError(f"No compatible installed Ollama model is available for role {role!r}.")

    def configured_roles(
        self,
        inventory: RuntimeInventory | None = None,
    ) -> dict[str, dict[str, Any]]:
        roles = [
            "planner",
            "simple",
            "general",
            "search",
            "reasoning",
            "fast_reasoning",
            "heavy",
            "synthesis",
            "writer",
            "classifier",
            "vision",
            "fallback",
        ]
        output: dict[str, dict[str, Any]] = {}
        for role in roles:
            configured = self.settings.model_for_key(role)
            try:
                resolved = self.resolve(role, inventory)
                error = None
            except RuntimeError as exc:
                resolved = None
                error = str(exc)
            output[role] = {
                "configured": configured,
                "resolved": resolved,
                "available": resolved is not None,
                "error": error,
            }

        available = set(inventory.model_names if inventory else [])
        output["embedding"] = {
            "configured": self.settings.embedding_model,
            "resolved": (
                self.settings.embedding_model
                if not available or self.settings.embedding_model in available
                else None
            ),
            "available": not available or self.settings.embedding_model in available,
            "chat_model": False,
        }
        return output

    def _compatible(self, role: str, model: str) -> bool:
        caps = capabilities_for(model)
        if role == "vision":
            return caps.vision
        if role == "embedding":
            return caps.embedding
        return caps.chat and not caps.embedding

    def _preferred_candidates(self, key: str) -> list[str]:
        preferences = {
            "planner": [
                self.settings.model_planner,
                self.settings.model_general,
                self.settings.model_classifier,
                self.settings.model_search,
            ],
            "simple": [
                self.settings.model_simple,
                self.settings.model_general,
                self.settings.model_fallback,
            ],
            "general": [
                self.settings.model_general,
                self.settings.model_simple,
                self.settings.model_search,
                self.settings.model_fallback,
            ],
            "search": [
                self.settings.model_search,
                self.settings.model_general,
                self.settings.model_fallback,
            ],
            "reasoning": [
                self.settings.model_reasoning,
                self.settings.model_fast_reasoning,
                self.settings.model_heavy,
                self.settings.model_search,
            ],
            "fast_reasoning": [
                self.settings.model_fast_reasoning,
                self.settings.model_reasoning,
                self.settings.model_general,
                self.settings.model_fallback,
            ],
            "heavy": [
                self.settings.model_heavy,
                self.settings.model_synthesis,
                self.settings.model_reasoning,
                self.settings.model_search,
            ],
            "synthesis": [
                self.settings.model_synthesis,
                self.settings.model_heavy,
                self.settings.model_reasoning,
                self.settings.model_search,
            ],
            "writer": [
                self.settings.model_writer,
                self.settings.model_general,
                self.settings.model_synthesis,
                self.settings.model_fallback,
            ],
            "classifier": [
                self.settings.model_classifier,
                self.settings.model_general,
                self.settings.model_simple,
                self.settings.model_fallback,
            ],
            "vision": [self.settings.model_vision],
            "fallback": [self.settings.model_fallback, self.settings.model_general],
        }
        merged: list[str] = []
        for candidate in preferences.get(
            key,
            [self.settings.model_for_key(key), self.settings.model_fallback],
        ):
            if candidate and candidate not in merged:
                merged.append(candidate)
        return merged


def normalize_inventory(value: RuntimeInventory | dict[str, Any]) -> RuntimeInventory:
    """Normalize legacy graph inventory dictionaries for a safe transition."""
    if isinstance(value, RuntimeInventory):
        return value
    return RuntimeInventory(
        models=[item for item in value.get("models", []) if isinstance(item, dict)],
        tools=[item for item in value.get("tools", []) if isinstance(item, dict)],
        errors={str(k): str(v) for k, v in value.get("errors", {}).items()},
        cached=bool(value.get("cached", False)),
        age_seconds=float(value.get("age_seconds", 0.0) or 0.0),
    )


def build_inventory_payload(
    settings: Settings,
    inventory: RuntimeInventory | dict[str, Any],
    selector: ModelSelector | Any,
) -> dict[str, Any]:
    runtime = normalize_inventory(inventory)
    effective_selector = selector if hasattr(selector, "configured_roles") else ModelSelector(settings)
    return {
        "service": {
            "name": settings.app_name,
            "environment": settings.environment,
            "backend": settings.llm_backend,
        },
        "ollama": {
            "base_url": settings.ollama_base_url,
            "models": runtime.models,
            "model_names": runtime.model_names,
            "configured_roles": effective_selector.configured_roles(runtime),
            "model_task_catalog": settings.model_role_catalog(),
        },
        "mcp": {
            "enabled": settings.mcp_enabled,
            "server_url": settings.mcp_server_url,
            "tools": runtime.tools,
            "tool_names": runtime.tool_names,
        },
        "cache": {
            "cached": runtime.cached,
            "age_seconds": round(runtime.age_seconds, 3),
        },
        "errors": runtime.errors,
    }


def _format_exception(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
