from __future__ import annotations


def after_budgeted_step(state: dict) -> str:
    """Route any budgeted node to safe termination when it set a reason."""
    return "terminate" if state.get("termination_reason") else "continue"


def after_verification(state: dict) -> str:
    # A budget or other controlled termination must take precedence over the
    # model-produced verdict. In particular, an incomplete task must never be
    # advanced merely because a legacy report used verdict="pass".
    if state.get("termination_reason"):
        return "terminate"

    verdict = (state.get("verification") or {}).get("verdict", "revise")
    if verdict == "pass":
        return "advance"
    if verdict == "research":
        return "research"
    if verdict == "replan":
        return "replan"
    return "revise"


def after_advance(state: dict) -> str:
    tasks = (state.get("plan") or {}).get("tasks", [])
    return "finalize" if state.get("task_index", 0) >= len(tasks) else "worker"
