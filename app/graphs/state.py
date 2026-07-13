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

    # Phase 5: conversation continuity is separate from checkpointed execution.
    conversation_id: str
    run_id: str
    execution_thread_id: str
    state_schema_version: int
    resume_requested: bool
    resumed: bool

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
    final_verification_required: bool
    final_verification: dict[str, Any]
    final_revision_rounds: int
