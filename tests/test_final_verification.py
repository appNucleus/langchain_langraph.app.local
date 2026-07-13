from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.graph as graph_module
from app.graph import ChatAgent
from app.schemas.execution import ExecutionBudget
from app.schemas.finalization import FinalVerificationReport
from app.schemas.verification import VerificationIssue
from app.settings import Settings


class FakeRouter:
    def select_model(self, **_kwargs):
        return SimpleNamespace(role="reasoning", model="verifier-model", reason="test")

    def execution_context(self):
        return {"current_date": "2026-07-12"}


class FakeFinalVerifier:
    def __init__(self, *_args, **_kwargs):
        pass

    async def verify_final(self, _payload):
        return FinalVerificationReport(
            verdict="revise",
            answer_complete=False,
            issues=[
                VerificationIssue(
                    code="unsupported_claim",
                    description="Candidate introduced a new claim",
                    severity="high",
                )
            ],
            required_actions=["Remove the unsupported claim"],
            confidence=0.9,
        )


def state() -> dict:
    return {
        "message": "Compare two APIs",
        "metadata": {},
        "request_id": "r1",
        "plan": {"tasks": [{"id": "t1"}, {"id": "t2"}]},
        "task_index": 2,
        "task_results": [
            {"worker_result": {"answer": "A"}, "verification": {"verdict": "pass"}},
            {"worker_result": {"answer": "B"}, "verification": {"verdict": "pass"}},
        ],
        "response": "A and B plus an invented statistic",
        "execution_budget": ExecutionBudget(60, 8, 4, 3),
        "inventory": {"models": [{"name": "verifier-model"}], "tools": []},
        "selected_models": {},
        "iterations": 0,
        "research_rounds": 0,
        "replans": 0,
        "final_revision_rounds": 1,
        "termination_reason": None,
    }


@pytest.mark.asyncio
async def test_failed_final_verification_prevents_completion(monkeypatch) -> None:
    monkeypatch.setattr(graph_module, "FinalVerifierAgent", FakeFinalVerifier)
    agent = ChatAgent.__new__(ChatAgent)
    agent.settings = Settings(
        llm_backend="ollama",
        final_max_revision_rounds=1,
    )
    agent.router = FakeRouter()

    result = await agent._verify_final(state())

    assert result["final_verification"]["verdict"] == "revise"
    assert result["termination_reason"] == "final answer could not be independently verified"
    assert agent._after_final_verification(result) == "terminate"


def test_final_verification_pass_routes_to_completion() -> None:
    agent = ChatAgent.__new__(ChatAgent)
    agent.settings = Settings(final_max_revision_rounds=1)
    current = state()
    current["final_verification"] = {
        "verdict": "pass",
        "answer_complete": True,
    }
    current["termination_reason"] = None

    assert agent._after_final_verification(current) == "complete"
