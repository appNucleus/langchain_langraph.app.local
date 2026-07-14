from __future__ import annotations

from typing import Any, TypedDict


class AgentGraphState(TypedDict, total=False):
    message: str
    system_prompt: str
    metadata: dict[str, Any]
    history: list[dict[str, Any]]

    conversation_id: str
    run_id: str
    execution_thread_id: str
    execution_meter_state: dict[str, Any]

    plan: dict[str, Any]
    task_index: int
    task_results: list[dict[str, Any]]
    worker_result: dict[str, Any]
    verification: dict[str, Any]
    evidence: list[dict[str, Any]]
    tool_errors: list[dict[str, Any]]
    grounding: list[dict[str, Any]]

    iterations: int
    research_rounds: int
    replans: int
    response: str
    backend: str
    model: str | None
    termination_reason: str
