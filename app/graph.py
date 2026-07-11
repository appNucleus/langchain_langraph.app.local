from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.agents.planner import PlannerAgent
from app.agents.synthesizer import SynthesizerAgent
from app.agents.verifier import VerifierAgent
from app.agents.worker import WorkerAgent
from app.graphs.routes import after_advance, after_verification
from app.graphs.state import AgentGraphState
from app.llm.ollama import OllamaClient
from app.mcp.client import MCPClient
from app.observability.events import event
from app.observability.metrics import metrics
from app.observability.tracing import span
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.execution import BudgetExceeded, ExecutionBudget
from app.schemas.planning import ExecutionPlan, PlanTask
from app.schemas.verification import VerificationIssue, VerificationReport
from app.schemas.worker import WorkerResult
from app.services.answer_quality import deterministic_output_issues
from app.services.context_builder import build_context
from app.services.evidence import evidence_from_metadata
from app.settings import Settings
from app.state import StateRuntime
from app.tools.executor import ToolApprovalRequired, ToolExecutor


def encode_sse(event_name: str, data: object) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class _Selector:
    def __init__(self, settings: Settings):
        self.settings = settings


class ChatAgent:
    """Phase 4 runtime with durable checkpoints and pluggable state backends."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.planner = PlannerAgent(settings, settings.model_planner)
        self.worker = WorkerAgent(settings, settings.model_general)
        self.verifier = VerifierAgent(settings, settings.model_reasoning)
        self.synthesizer = SynthesizerAgent(settings, settings.model_synthesis)
        self.ollama = OllamaClient(settings)
        self.mcp = MCPClient(settings)
        self.tool_executor = ToolExecutor(self.mcp, settings)
        self.selector = _Selector(settings)
        self.state_runtime = StateRuntime(settings)
        self.memory = self.state_runtime.conversations
        self.graph = self._build_graph(self.state_runtime.checkpointer)

    async def start(self) -> None:
        await self.state_runtime.start()
        self.memory = self.state_runtime.conversations
        self.graph = self._build_graph(self.state_runtime.checkpointer)
        if self.settings.llm_backend == "ollama":
            await self.ollama.start()
        if self.settings.mcp_enabled:
            await self.mcp.start()

    async def aclose(self) -> None:
        await asyncio.gather(
            self.ollama.aclose(),
            self.mcp.aclose(),
            self.state_runtime.aclose(),
            return_exceptions=True,
        )

    async def persistence_health(self) -> dict[str, Any]:
        return await self.state_runtime.health()

    async def load_inventory(self) -> dict[str, object]:
        models: list[dict[str, Any]] = []
        tools: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        if self.settings.llm_backend == "ollama":
            try:
                models = await self.ollama.list_models()
            except Exception as exc:  # noqa: BLE001
                errors["ollama"] = str(exc)
        if self.settings.mcp_enabled:
            try:
                tools = await self.mcp.list_tools()
            except Exception as exc:  # noqa: BLE001
                errors["mcp"] = str(exc)
        return {"models": models, "tools": tools, "errors": errors}

    def _build_graph(self, checkpointer: Any):
        builder = StateGraph(AgentGraphState)
        builder.add_node("plan", self._plan)
        builder.add_node("worker", self._worker)
        builder.add_node("verify", self._verify)
        builder.add_node("revise", self._revise)
        builder.add_node("research", self._research)
        builder.add_node("replan", self._replan)
        builder.add_node("advance", self._advance)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "worker")
        builder.add_edge("worker", "verify")
        builder.add_conditional_edges(
            "verify",
            after_verification,
            {
                "advance": "advance",
                "revise": "revise",
                "research": "research",
                "replan": "replan",
            },
        )
        builder.add_edge("revise", "verify")
        builder.add_edge("research", "worker")
        builder.add_edge("replan", "worker")
        builder.add_conditional_edges(
            "advance", after_advance, {"worker": "worker", "finalize": "finalize"}
        )
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=checkpointer)

    @staticmethod
    def _budget(state: AgentGraphState) -> ExecutionBudget:
        value = state.get("execution_budget")
        if isinstance(value, ExecutionBudget):
            return value
        raise RuntimeError("execution budget is missing")

    async def _plan(self, state: AgentGraphState) -> AgentGraphState:
        budget = self._budget(state)
        budget.check()
        if self.settings.llm_backend != "ollama":
            plan = ExecutionPlan(
                goal=state["message"],
                tasks=[
                    PlanTask(
                        id="t1",
                        objective=state["message"],
                        completion_criteria=["Answer directly and completely"],
                    )
                ],
            )
        else:
            budget.model_calls += 1
            budget.check()
            with span("graph.plan"):
                plan = await self.planner.plan(state["message"])
        return {
            **state,
            "plan": plan.model_dump(),
            "task_index": 0,
            "task_results": [],
            "iterations": 0,
            "research_rounds": 0,
            "replans": 0,
        }

    def _current_task(self, state: AgentGraphState) -> dict[str, Any]:
        return state["plan"]["tasks"][state.get("task_index", 0)]

    async def _worker(self, state: AgentGraphState) -> AgentGraphState:
        budget = self._budget(state)
        budget.check()
        task = self._current_task(state)
        evidence = evidence_from_metadata(state.get("metadata", {}))
        context = build_context(evidence, self.settings.phase2_max_context_chars)
        payload = {
            "user_request": state["message"],
            "task": task,
            "evidence": context,
            "history": state.get("history", []),
            "previous_verification": state.get("verification"),
        }
        if self.settings.llm_backend != "ollama":
            result = WorkerResult(
                answer=f"Echo mode is active.\n\nMessage received: {state['message']}",
                confidence=0.5,
            )
        else:
            budget.model_calls += 1
            budget.check()
            with span("graph.worker"):
                result = await self.worker.execute(payload)
        return {
            **state,
            "worker_result": result.model_dump(),
            "evidence": [item.model_dump() for item in evidence],
            "iterations": state.get("iterations", 0) + 1,
        }

    async def _verify(self, state: AgentGraphState) -> AgentGraphState:
        budget = self._budget(state)
        budget.verifier_rounds += 1
        try:
            budget.check()
        except BudgetExceeded as exc:
            report = VerificationReport(
                verdict="pass",
                task_complete=False,
                issues=[
                    VerificationIssue(
                        code="budget_exhausted", description=str(exc), severity="high"
                    )
                ],
                required_actions=[],
                confidence=0.2,
            )
            return {
                **state,
                "verification": report.model_dump(),
                "termination_reason": str(exc),
            }
        issues = deterministic_output_issues(state["worker_result"]["answer"])
        if self.settings.llm_backend != "ollama":
            report = VerificationReport(
                verdict="pass",
                task_complete=not issues,
                issues=[VerificationIssue(code=item, description=item) for item in issues],
                confidence=0.5,
            )
        else:
            budget.model_calls += 1
            try:
                budget.check()
            except BudgetExceeded as exc:
                report = VerificationReport(
                    verdict="pass",
                    task_complete=False,
                    issues=[
                        VerificationIssue(
                            code="budget_exhausted",
                            description=str(exc),
                            severity="high",
                        )
                    ],
                    required_actions=[],
                    confidence=0.2,
                )
                return {
                    **state,
                    "verification": report.model_dump(),
                    "termination_reason": str(exc),
                }
            with span("graph.verify"):
                report = await self.verifier.verify(
                    {
                        "user_request": state["message"],
                        "task": self._current_task(state),
                        "worker_result": state["worker_result"],
                        "evidence": state.get("evidence", []),
                        "deterministic_issues": issues,
                    }
                )
        return {**state, "verification": report.model_dump()}

    async def _revise(self, state: AgentGraphState) -> AgentGraphState:
        budget = self._budget(state)
        budget.model_calls += 1
        budget.check()
        payload = {
            "user_request": state["message"],
            "task": self._current_task(state),
            "worker_result": state["worker_result"],
            "verification": state["verification"],
            "evidence": state.get("evidence", []),
        }
        result = (
            await self.worker.revise(payload)
            if self.settings.llm_backend == "ollama"
            else WorkerResult.model_validate(state["worker_result"])
        )
        return {
            **state,
            "worker_result": result.model_dump(),
            "iterations": state.get("iterations", 0) + 1,
        }

    async def _research(self, state: AgentGraphState) -> AgentGraphState:
        rounds = state.get("research_rounds", 0) + 1
        if rounds > self.settings.phase2_max_research_rounds or not self.settings.mcp_enabled:
            verification = dict(state["verification"])
            verification["verdict"] = "revise"
            return {**state, "verification": verification, "research_rounds": rounds}
        metadata = dict(state.get("metadata", {}))
        query = str(
            metadata.get("research_query")
            or self._current_task(state).get("objective")
            or state["message"]
        )
        tool = str(metadata.get("research_tool") or "web_search_and_scrape")
        arguments = {
            "query": query,
            "pages": int(metadata.get("pages", 3)),
            "prefer_official": True,
        }
        try:
            result = await self.tool_executor.execute(
                tool, arguments, budget=self._budget(state), metadata=metadata
            )
            evidence = metadata.setdefault("evidence", [])
            evidence.append(
                {
                    "id": f"research-{rounds}",
                    "source": tool,
                    "content": result.data if result.ok else f"Tool failed: {result.error}",
                }
            )
        except (ToolApprovalRequired, BudgetExceeded) as exc:
            metadata.setdefault("evidence", []).append(
                {"id": f"research-{rounds}", "source": tool, "content": str(exc)}
            )
        return {**state, "metadata": metadata, "research_rounds": rounds}

    async def _replan(self, state: AgentGraphState) -> AgentGraphState:
        replans = state.get("replans", 0) + 1
        if replans > self.settings.phase2_max_replans:
            verification = dict(state["verification"])
            verification["verdict"] = "revise"
            return {**state, "verification": verification, "replans": replans}
        budget = self._budget(state)
        budget.model_calls += 1
        budget.check()
        plan = (
            await self.planner.plan(state["message"])
            if self.settings.llm_backend == "ollama"
            else ExecutionPlan.model_validate(state["plan"])
        )
        return {
            **state,
            "plan": plan.model_dump(),
            "task_index": 0,
            "replans": replans,
        }

    async def _advance(self, state: AgentGraphState) -> AgentGraphState:
        results = list(state.get("task_results", []))
        results.append(
            {
                "task": self._current_task(state),
                "worker_result": state["worker_result"],
                "verification": state["verification"],
            }
        )
        return {
            **state,
            "task_results": results,
            "task_index": state.get("task_index", 0) + 1,
            "verification": {},
            "worker_result": {},
        }

    async def _finalize(self, state: AgentGraphState) -> AgentGraphState:
        results = state.get("task_results", [])
        if self.settings.llm_backend == "ollama" and len(results) > 1:
            budget = self._budget(state)
            budget.model_calls += 1
            budget.check()
            response = await self.synthesizer.synthesize(
                {"user_request": state["message"], "verified_results": results}
            )
        else:
            response = (
                results[-1]["worker_result"]["answer"]
                if results
                else state.get("worker_result", {}).get("answer", "")
            )
        return {
            **state,
            "response": response,
            "backend": self.settings.llm_backend,
            "model": self.settings.model_general,
        }

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        thread_id = request.thread_id or str(uuid4())
        budget = ExecutionBudget(
            self.settings.execution_max_duration_seconds,
            self.settings.execution_max_model_calls,
            self.settings.execution_max_tool_calls,
            self.settings.execution_max_verifier_rounds,
        )
        history = await self.memory.get(thread_id)
        metrics.inc("chat.requests")
        config = {"configurable": {"thread_id": thread_id}}
        with span("chat.total"):
            result = await self.graph.ainvoke(
                {
                    "message": request.message,
                    "system_prompt": request.system_prompt
                    or self.settings.default_system_prompt,
                    "metadata": request.metadata,
                    "history": history,
                    "execution_budget": budget,
                },
                config=config,
            )
        await self.memory.append(
            thread_id,
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": result["response"]},
        )
        return ChatResponse.from_result(
            thread_id=thread_id,
            response=result["response"],
            backend=result["backend"],
            model=result.get("model"),
            metadata={
                **request.metadata,
                "phase": "4",
                "persistence": {
                    "state_backend": self.settings.state_backend,
                    "checkpoint_backend": self.settings.checkpoint_backend,
                    "artifact_backend": self.settings.artifact_backend,
                },
                "plan": result.get("plan"),
                "verification": result.get("task_results"),
                "iterations": result.get("iterations"),
                "termination_reason": result.get("termination_reason"),
                "usage": {
                    "model_calls": budget.model_calls,
                    "tool_calls": budget.tool_calls,
                    "verifier_rounds": budget.verifier_rounds,
                    "elapsed_seconds": round(budget.elapsed_seconds, 3),
                },
            },
        )

    async def astream_events(
        self, request: ChatRequest
    ) -> AsyncIterator[dict[str, object]]:
        yield event("request_started", thread_id=request.thread_id)
        yield event("planning_started")
        task = asyncio.create_task(self.ainvoke(request))
        while not task.done():
            await asyncio.sleep(0.5)
            yield event("working", stage="agent_graph")
        response = await task
        yield event("completed", response=response.model_dump())
