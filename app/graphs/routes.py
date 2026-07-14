from __future__ import annotations

from typing import Any


def after_budgeted_step(state: dict[str, Any]) -> str:
    """Stop immediately when a prior node recorded a terminal condition."""

    return "terminate" if state.get("termination_reason") else "continue"


def after_plan(state: dict[str, Any]) -> str:
    """Route the first planned task without bypassing pre-research decisions."""

    if state.get("termination_reason"):
        return "terminate"
    return "research" if state.get("next_action") == "research" else "worker"


def after_verification(state: dict[str, Any]) -> str:
    """Fail closed and advance only a complete, explicitly passing task."""

    if state.get("termination_reason"):
        return "terminate"

    verification = state.get("verification") or {}
    verdict = str(verification.get("verdict", "terminate"))
    if verdict == "pass":
        return "advance" if verification.get("task_complete") is True else "revise"
    if verdict in {"revise", "research", "replan", "terminate"}:
        return verdict
    return "terminate"


def after_advance(state: dict[str, Any]) -> str:
    """Route the next task, preserving the existing research-first behavior."""

    if state.get("termination_reason"):
        return "terminate"

    task_index = int(state.get("task_index", 0))
    tasks = (state.get("plan") or {}).get("tasks", [])
    if task_index >= len(tasks):
        return "finalize"
    return "research" if state.get("next_action") == "research" else "worker"
