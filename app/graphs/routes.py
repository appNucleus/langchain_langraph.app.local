from __future__ import annotations


def after_budgeted_step(state: dict) -> str:
    return "terminate" if state.get("termination_reason") else "continue"


def after_plan(state: dict) -> str:
    if state.get("termination_reason"):
        return "terminate"
    return "research" if state.get("next_action") == "research" else "worker"


def after_verification(state: dict) -> str:
    if state.get("termination_reason"):
        return "terminate"
    verification = state.get("verification") or {}
    verdict = verification.get("verdict", "revise")
    if verdict == "pass" and verification.get("task_complete") is True:
        return "advance"
    if verdict == "research":
        return "research"
    if verdict == "replan":
        return "replan"
    return "revise"


def after_advance(state: dict) -> str:
    tasks = (state.get("plan") or {}).get("tasks", [])
    if state.get("task_index", 0) >= len(tasks):
        return "finalize"
    return "research" if state.get("next_action") == "research" else "worker"
