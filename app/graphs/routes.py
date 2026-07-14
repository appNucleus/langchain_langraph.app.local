from __future__ import annotations

from typing import Any


def after_verification(state: dict[str, Any]) -> str:
    verdict = str(state.get("verification", {}).get("verdict", "terminate"))
    if verdict == "pass":
        return "advance"
    if verdict in {"revise", "research", "replan", "terminate"}:
        return verdict
    return "terminate"


def after_advance(state: dict[str, Any]) -> str:
    task_index = int(state.get("task_index", 0))
    tasks = state.get("plan", {}).get("tasks", [])
    return "worker" if task_index < len(tasks) else "finalize"
