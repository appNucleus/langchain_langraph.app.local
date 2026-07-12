from __future__ import annotations

from typing import Any, TypedDict

from app.schemas.execution import ExecutionBudget


class AgentGraphState(TypedDict, total=False):
    """Complete mutable state shared by all LangGraph nodes.

    Every value supplied to ``StateGraph.ainvoke()`` and later read by a graph
    node must be declared here. LangGraph builds its state channels from this
    schema; undeclared input keys are not available to nodes.
    """

    message: str
    system_prompt: str
    metadata: dict[str, Any]
    history: list[dict[str, Any]]
    execution_budget: ExecutionBudget

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
