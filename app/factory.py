from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.exception_handlers import register_exception_handlers
from app.api.openapi import (
    install_openapi_customization,
    openapi_with_chat_request_example,
)
from app.api.router import router as api_router
from app.core.lifespan import build_lifespan
from app.graph import ChatAgent
from app.logging_config import configure_logging
from app.orchestration.chat_runtime import ChatRuntimeAgent
from app.orchestration.execution_meter import execution_meter_scope
from app.schemas.chat import load_chat_request_example
from app.schemas.execution import ExecutionBudget
from app.services.inventory import InventoryService
from app.settings import Settings, get_settings


class _MeteredChatRuntimeAgent(ChatRuntimeAgent):
    """Keep the runtime-only meter out of durable LangGraph state."""

    async def _invoke_graph_safely(
        self,
        *,
        graph_input: Any,
        config: dict[str, Any],
        identity: Any,
        request_id: str,
        budget: ExecutionBudget,
    ) -> dict[str, Any]:
        if isinstance(graph_input, dict):
            graph_input = self._checkpoint_safe_state(graph_input, budget)
        with execution_meter_scope(budget):
            return await super()._invoke_graph_safely(
                graph_input=graph_input,
                config=config,
                identity=identity,
                request_id=request_id,
                budget=budget,
            )


def create_app(
    *,
    settings: Settings | None = None,
    chat_agent: ChatAgent | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)
    agent = chat_agent or _MeteredChatRuntimeAgent(app_settings)
    inventory_service = getattr(agent, "inventory_service", None)
    if inventory_service is None:
        inventory_service = InventoryService(app_settings, agent.ollama, agent.mcp)

    app = FastAPI(
        title=app_settings.app_name,
        version=__version__,
        description=(
            "FastAPI + LangGraph local assistant with bounded execution, "
            "durable run outcomes, checkpoints, Ollama model routing, and MCP tools."
        ),
        lifespan=build_lifespan(app_settings, agent),
    )
    app.state.settings = app_settings
    app.state.chat_agent = agent
    app.state.inventory_service = inventory_service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(api_router)
    install_openapi_customization(
        app,
        request_example_loader=load_chat_request_example,
    )
    return app


__all__ = [
    "create_app",
    "load_chat_request_example",
    "openapi_with_chat_request_example",
]
