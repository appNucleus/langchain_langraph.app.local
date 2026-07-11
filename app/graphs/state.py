from __future__ import annotations
from typing import Any, TypedDict

class AgentGraphState(TypedDict, total=False):
    message: str
    system_prompt: str
    metadata: dict[str, Any]
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
