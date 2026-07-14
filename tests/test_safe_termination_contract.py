from __future__ import annotations

import ast
from pathlib import Path

from app.graphs.routes import after_verification


def test_unknown_or_explicit_termination_routes_fail_closed() -> None:
    assert after_verification({"verification": {"verdict": "terminate"}}) == "terminate"
    assert after_verification({"verification": {"verdict": "unexpected"}}) == "terminate"


def test_budget_exhaustion_is_never_constructed_as_pass() -> None:
    graph_path = Path(__file__).resolve().parents[1] / "app" / "graph.py"
    tree = ast.parse(graph_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "VerificationReport":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        verdict = keywords.get("verdict")
        complete = keywords.get("task_complete")
        assert not (
            isinstance(verdict, ast.Constant)
            and verdict.value == "pass"
            and isinstance(complete, ast.Constant)
            and complete.value is False
        )
