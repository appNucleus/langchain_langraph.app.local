from __future__ import annotations

from typing import Any, TypedDict

from app.schemas.execution import ExecutionBudget


class AgentGraphState(TypedDict, total=False):
    message: str
    system_prompt: str
    system_prompt_source: str
    request_domain: str
    metadata: dict[str, Any]
    history: list[dict[str, Any]]
    execution_budget: ExecutionBudget
    request_id: str

    inventory: dict[str, Any]
    routing: dict[str, Any]
    selected_models: dict[str, str]
    selected_tool: str | None
    selected_tools: dict[str, Any]
    researched_task_ids: list[str]
    research_queries: dict[str, list[str]]

    plan: dict[str, Any]
    task_index: int
    task_results: list[dict[str, Any]]
    worker_result: dict[str, Any]
    verification: dict[str, Any]
    evidence: list[dict[str, Any]]
    iterations: int
    research_rounds: int
    replans: int
    next_action: str
    response: str
    backend: str
    model: str | None
    termination_reason: str | None
