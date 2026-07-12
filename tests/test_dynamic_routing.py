from __future__ import annotations

from app.graphs.routes import after_advance, after_plan
from app.services.inventory import RuntimeInventory
from app.services.routing import RuntimeRouter, build_fresh_run_state
from app.settings import Settings


def _settings() -> Settings:
    return Settings(
        llm_backend="ollama",
        mcp_enabled=True,
        model_planner="qwen3.5:4b",
        model_simple="qwen3.5:2b",
        model_general="qwen3.5:4b",
        model_search="qwen3.5:9b",
        model_reasoning="deepseek-r1:8b",
        model_fast_reasoning="phi4-mini-reasoning:latest",
        model_synthesis="gemma4:12b-it-qat",
        model_fallback="granite3.3:8b",
    )


def _inventory() -> RuntimeInventory:
    return RuntimeInventory(
        models=[
            {"name": "qwen3.5:2b"},
            {"name": "qwen3.5:4b"},
            {"name": "qwen3.5:9b"},
            {"name": "deepseek-r1:8b"},
            {"name": "phi4-mini-reasoning:latest"},
            {"name": "gemma4:12b-it-qat"},
            {"name": "granite3.3:8b"},
        ],
        tools=[
            {
                "name": "web_search_and_scrape",
                "description": "Search the web and scrape relevant pages",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "pages": {"type": "integer"},
                        "prefer_official": {"type": "boolean"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "sports_news_search",
                "description": "Search current football, soccer, FIFA and sports news",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "mail_send_draft",
                "description": "Send a mail draft",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ],
    )


def test_sports_yesterday_routes_to_external_search_and_reasoning_verifier() -> None:
    router = RuntimeRouter(_settings())
    decision = router.classify_task(
        user_request=(
            "Can you find yesterday's FIFA World Cup game against Argentina "
            "and analyze criticism of the red card?"
        ),
        task={
            "id": "t1",
            "objective": "Find the match and red-card criticism",
            "required_evidence": ["current match reports and criticism"],
        },
    )

    assert decision.requires_external_evidence is True
    assert decision.worker_role == "search"
    assert decision.verifier_role == "reasoning"

    selected = router.select_model(
        role=decision.worker_role,
        inventory=_inventory(),
        reason=decision.reason,
    )
    assert selected.model == "qwen3.5:9b"


def test_dynamic_tool_selection_prefers_live_sports_tool_and_excludes_write_tool() -> None:
    router = RuntimeRouter(_settings())
    decisions = router.select_tools(
        user_request=(
            "Find yesterday's FIFA World Cup game against Argentina and the "
            "criticism of a red card"
        ),
        task={"objective": "Locate current reports about the match and red card"},
        required_actions=["Find reputable match reports and expert criticism"],
        inventory=_inventory(),
        metadata={"pages": 4},
    )

    assert decisions
    assert decisions[0].name == "sports_news_search"
    assert decisions[0].arguments["max_results"] == 4
    assert "mail_send_draft" not in {item.name for item in decisions}


def test_generic_current_research_uses_web_search_fallback() -> None:
    router = RuntimeRouter(_settings())
    inventory = RuntimeInventory(
        models=_inventory().models,
        tools=[_inventory().tools[0]],
    )
    decisions = router.select_tools(
        user_request="Find the latest documented API changes",
        task={"objective": "Research the latest official documentation"},
        inventory=inventory,
        metadata={},
    )

    assert [item.name for item in decisions] == ["web_search_and_scrape"]
    assert decisions[0].arguments == {
        "query": (
            "Research the latest official documentation "
            "Find the latest documented API changes"
        ),
        "pages": 3,
        "prefer_official": True,
    }


def test_fresh_run_state_explicitly_clears_all_stale_checkpoint_channels() -> None:
    sentinel_budget = object()
    state = build_fresh_run_state(
        message="new sports request",
        system_prompt="sports analyst",
        metadata={"safe_read_only_test": True},
        history=[{"role": "assistant", "content": "old weather answer"}],
        execution_budget=sentinel_budget,
        request_id="request-1",
        inventory=_inventory(),
        backend="ollama",
    )

    assert state["message"] == "new sports request"
    assert state["history"]
    assert state["execution_budget"] is sentinel_budget
    assert state["plan"] == {}
    assert state["task_results"] == []
    assert state["worker_result"] == {}
    assert state["verification"] == {}
    assert state["evidence"] == []
    assert state["termination_reason"] is None
    assert state["response"] == ""
    assert state["selected_tool"] is None
    assert state["selected_tools"] == {}
    assert state["researched_task_ids"] == []


def test_routes_pre_research_and_next_task_research() -> None:
    state = {
        "termination_reason": None,
        "next_action": "research",
        "task_index": 0,
        "plan": {"tasks": [{"id": "t1"}, {"id": "t2"}]},
    }
    assert after_plan(state) == "research"

    state["task_index"] = 1
    assert after_advance(state) == "research"

    state["task_index"] = 2
    assert after_advance(state) == "finalize"

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graphs.state import AgentGraphState


@pytest.mark.asyncio
async def test_fresh_state_overwrites_stale_langgraph_checkpoint_for_same_thread() -> None:
    seen: list[dict] = []

    async def capture(state: AgentGraphState) -> AgentGraphState:
        seen.append(dict(state))
        if state["message"] == "old weather request":
            return {
                **state,
                "plan": {"tasks": [{"id": "weather"}]},
                "worker_result": {"answer": "old weather answer"},
                "termination_reason": "maximum verifier rounds exceeded",
                "response": "old weather answer",
            }
        return {**state, "response": "new sports answer"}

    builder = StateGraph(AgentGraphState)
    builder.add_node("capture", capture)
    builder.add_edge(START, "capture")
    builder.add_edge("capture", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "reused-thread"}}

    old_state = build_fresh_run_state(
        message="old weather request",
        system_prompt="weather",
        metadata={},
        history=[],
        execution_budget="budget-1",
        request_id="request-1",
        inventory=_inventory(),
        backend="ollama",
    )
    await graph.ainvoke(old_state, config=config)

    new_state = build_fresh_run_state(
        message="new sports request",
        system_prompt="sports",
        metadata={},
        history=[{"role": "assistant", "content": "old weather answer"}],
        execution_budget="budget-2",
        request_id="request-2",
        inventory=_inventory(),
        backend="ollama",
    )
    result = await graph.ainvoke(new_state, config=config)

    assert len(seen) == 2
    incoming_new = seen[1]
    assert incoming_new["message"] == "new sports request"
    assert incoming_new["plan"] == {}
    assert incoming_new["worker_result"] == {}
    assert incoming_new["termination_reason"] is None
    assert incoming_new["response"] == ""
    assert result["response"] == "new sports answer"

from types import SimpleNamespace

from app.graph import ChatAgent
from app.schemas.chat import ChatRequest
from app.schemas.planning import ExecutionPlan, PlanTask
from app.schemas.verification import VerificationReport
from app.schemas.worker import Claim, WorkerResult


@pytest.mark.asyncio
async def test_full_graph_reuses_conversation_thread_without_reusing_old_run_state(monkeypatch, caplog) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="app.graph")
    settings = _settings()
    agent = ChatAgent(settings)
    inventory = _inventory()

    class FakeInventoryService:
        async def load(self):
            return inventory

    used_models: list[tuple[str, str]] = []
    used_tools: list[str] = []

    class FakePlannerAgent:
        def __init__(self, _settings, model):
            used_models.append(("planner", model))

        async def invoke_json(self, *, system, payload, schema):
            request = payload["request"].lower()
            if "sports" in request or "fifa" in request or "argentina" in request:
                return ExecutionPlan(
                    goal="Find the requested current sports event and red-card criticism",
                    tasks=[
                        PlanTask(
                            id="sports-task",
                            objective="Find current FIFA match reports and red-card criticism",
                            required_evidence=["current sports reports"],
                            completion_criteria=["Evidence-backed sports answer"],
                        )
                    ],
                )
            return ExecutionPlan(
                goal="Find current weather",
                tasks=[
                    PlanTask(
                        id="weather-task",
                        objective="Find current Indianapolis weather",
                        required_evidence=["current weather evidence"],
                        completion_criteria=["Evidence-backed weather answer"],
                    )
                ],
            )

    class FakeWorkerAgent:
        def __init__(self, _settings, model):
            self.model = model
            used_models.append(("worker", model))

        async def execute(self, payload):
            evidence = payload["evidence"]
            assert evidence, "pre-research must supply evidence before worker execution"
            objective = payload["task"]["objective"]
            answer = (
                "SPORTS RESULT from current MCP evidence"
                if "FIFA" in objective or "sports" in objective.lower()
                else "WEATHER RESULT from current MCP evidence"
            )
            return WorkerResult(
                answer=answer,
                claims=[Claim(text=answer, evidence_ids=[evidence[0]["id"]])],
                confidence=0.9,
            )

        async def revise(self, payload):
            return WorkerResult.model_validate(payload["worker_result"])

    class FakeVerifierAgent:
        def __init__(self, _settings, model):
            used_models.append(("verifier", model))

        async def verify(self, payload):
            assert payload["evidence"]
            return VerificationReport(
                verdict="pass",
                task_complete=True,
                confidence=0.95,
            )

    class FakeSynthesizerAgent:
        def __init__(self, _settings, model):
            used_models.append(("synthesis", model))

        async def synthesize(self, payload):
            return "SYNTHESIZED"

    class FakeToolExecutor:
        async def execute(self, name, arguments, *, budget, metadata):
            used_tools.append(name)
            budget.tool_calls += 1
            return SimpleNamespace(
                ok=True,
                data={"source": name, "query": arguments.get("query"), "items": ["evidence"]},
                error=None,
            )

    monkeypatch.setattr("app.graph.PlannerAgent", FakePlannerAgent)
    monkeypatch.setattr("app.graph.WorkerAgent", FakeWorkerAgent)
    monkeypatch.setattr("app.graph.VerifierAgent", FakeVerifierAgent)
    monkeypatch.setattr("app.graph.SynthesizerAgent", FakeSynthesizerAgent)
    agent.inventory_service = FakeInventoryService()
    agent.tool_executor = FakeToolExecutor()

    first = await agent.ainvoke(
        ChatRequest(
            message="Find current weather in Indianapolis",
            thread_id="reused-thread",
            system_prompt="Act as a weather analyst",
        )
    )
    assert "WEATHER RESULT" in first.response

    second = await agent.ainvoke(
        ChatRequest(
            message=(
                "Can you find yesterday's FIFA World Cup game against Argentina "
                "and find criticism of the red card?"
            ),
            thread_id="reused-thread",
            system_prompt="Act as a sports analyst",
        )
    )

    assert "SPORTS RESULT" in second.response
    assert "WEATHER RESULT" not in second.response
    assert second.metadata["termination_reason"] is None
    assert second.metadata["selected_models"]["worker:sports-task"] == "qwen3.5:9b"
    assert second.metadata["selected_models"]["verify:sports-task"] == "deepseek-r1:8b"
    assert second.metadata["selected_tools"]["sports-task"] == "sports_news_search"
    assert used_tools[-1] == "sports_news_search"

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "graph_runtime_inventory" in messages
    assert "graph_task_route" in messages
    assert "graph_model_selected" in messages
    assert "graph_tool_candidates" in messages
    assert "graph_tool_selected" in messages
    assert "transient_state_reset=True" in messages
    assert "old weather answer" not in messages
