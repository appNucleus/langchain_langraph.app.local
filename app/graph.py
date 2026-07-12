from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.agents.planner import PlannerAgent
from app.agents.synthesizer import SynthesizerAgent
from app.agents.verifier import VerifierAgent
from app.agents.worker import WorkerAgent
from app.graphs.routes import after_advance, after_budgeted_step, after_verification
from app.graphs.state import AgentGraphState
from app.llm.ollama import OllamaClient
from app.logging_config import log_kv
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

logger = logging.getLogger(__name__)


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
        builder.add_node("terminate", self._terminate)

        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan",
            after_budgeted_step,
            {"continue": "worker", "terminate": "terminate"},
        )
        builder.add_conditional_edges(
            "worker",
            after_budgeted_step,
            {"continue": "verify", "terminate": "terminate"},
        )
        builder.add_conditional_edges(
            "verify",
            after_verification,
            {
                "advance": "advance",
                "revise": "revise",
                "research": "research",
                "replan": "replan",
                "terminate": "terminate",
            },
        )
        builder.add_conditional_edges(
            "revise",
            after_budgeted_step,
            {"continue": "verify", "terminate": "terminate"},
        )
        builder.add_conditional_edges(
            "research",
            after_budgeted_step,
            {"continue": "worker", "terminate": "terminate"},
        )
        builder.add_conditional_edges(
            "replan",
            after_budgeted_step,
            {"continue": "worker", "terminate": "terminate"},
        )
        builder.add_conditional_edges(
            "advance", after_advance, {"worker": "worker", "finalize": "finalize"}
        )
        builder.add_edge("finalize", END)
        builder.add_edge("terminate", END)
        return builder.compile(checkpointer=checkpointer)

    @staticmethod
    def _budget(state: AgentGraphState) -> ExecutionBudget:
        value = state.get("execution_budget")
        if isinstance(value, ExecutionBudget):
            return value
        raise RuntimeError("execution budget is missing")

    @staticmethod
    def _budget_fields(budget: ExecutionBudget) -> dict[str, object]:
        return {
            "elapsed_seconds": round(budget.elapsed_seconds, 3),
            "max_duration_seconds": budget.max_duration_seconds,
            "model_calls": budget.model_calls,
            "max_model_calls": budget.max_model_calls,
            "tool_calls": budget.tool_calls,
            "max_tool_calls": budget.max_tool_calls,
            "verifier_rounds": budget.verifier_rounds,
            "max_verifier_rounds": budget.max_verifier_rounds,
        }

    @staticmethod
    def _task_fields(state: AgentGraphState) -> dict[str, object]:
        plan = state.get("plan") or {}
        tasks = plan.get("tasks") or []
        index = int(state.get("task_index", 0))
        task_id: object | None = None
        if 0 <= index < len(tasks) and isinstance(tasks[index], dict):
            task_id = tasks[index].get("id")
        metadata = state.get("metadata") or {}
        return {
            "run_id": metadata.get("run_id"),
            "task_index": index,
            "task_count": len(tasks),
            "task_id": task_id,
            "iterations": state.get("iterations", 0),
            "research_rounds": state.get("research_rounds", 0),
            "replans": state.get("replans", 0),
        }

    def _log_node(
        self,
        level: int,
        event_name: str,
        *,
        node: str,
        state: AgentGraphState,
        **fields: object,
    ) -> None:
        budget = self._budget(state)
        log_kv(
            logger,
            level,
            event_name,
            node=node,
            **self._task_fields(state),
            **self._budget_fields(budget),
            **fields,
        )

    def _budget_termination(
        self,
        state: AgentGraphState,
        *,
        node: str,
        exc: BudgetExceeded,
        verification: VerificationReport | None = None,
    ) -> AgentGraphState:
        metrics.inc("graph.budget_exhausted")
        self._log_node(
            logging.WARNING,
            "graph_budget_exhausted",
            node=node,
            state=state,
            reason=str(exc),
        )
        update: AgentGraphState = {
            **state,
            "termination_reason": str(exc),
        }
        if verification is not None:
            update["verification"] = verification.model_dump()
        return update

    def _check_budget(
        self, state: AgentGraphState, *, node: str
    ) -> tuple[ExecutionBudget, AgentGraphState | None]:
        budget = self._budget(state)
        try:
            budget.check()
        except BudgetExceeded as exc:
            return budget, self._budget_termination(state, node=node, exc=exc)
        return budget, None

    @staticmethod
    def _partial_response(state: AgentGraphState) -> str:
        reason = str(state.get("termination_reason") or "execution stopped")
        sections: list[str] = []

        for index, item in enumerate(state.get("task_results", []), start=1):
            if not isinstance(item, dict):
                continue
            worker_result = item.get("worker_result") or {}
            answer = str(worker_result.get("answer") or "").strip()
            if answer:
                sections.append(f"Completed task {index}:\n{answer}")

        current = state.get("worker_result") or {}
        current_answer = str(current.get("answer") or "").strip()
        if current_answer and all(current_answer not in section for section in sections):
            sections.append(
                "In-progress task output (not fully verified):\n" + current_answer
            )

        if sections:
            body = "\n\n".join(sections)
            return (
                f"{body}\n\n"
                f"Execution stopped safely before all work was completed: {reason}. "
                "Treat any in-progress output as unverified."
            )
        return (
            "Execution stopped safely before a verified answer could be produced: "
            f"{reason}."
        )

    async def _plan(self, state: AgentGraphState) -> AgentGraphState:
        self._log_node(logging.INFO, "graph_node_start", node="plan", state=state)
        budget, terminated = self._check_budget(state, node="plan:precheck")
        if terminated:
            return terminated

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
            _, terminated = self._check_budget(state, node="plan:model_call")
            if terminated:
                return terminated
            with span("graph.plan"):
                plan = await self.planner.plan(state["message"])

        result: AgentGraphState = {
            **state,
            "plan": plan.model_dump(),
            "task_index": 0,
            "task_results": [],
            "iterations": 0,
            "research_rounds": 0,
            "replans": 0,
        }
        self._log_node(
            logging.INFO,
            "graph_node_complete",
            node="plan",
            state=result,
            planned_tasks=len(plan.tasks),
        )
        return result

    def _current_task(self, state: AgentGraphState) -> dict[str, Any]:
        return state["plan"]["tasks"][state.get("task_index", 0)]

    async def _worker(self, state: AgentGraphState) -> AgentGraphState:
        self._log_node(logging.INFO, "graph_node_start", node="worker", state=state)
        budget, terminated = self._check_budget(state, node="worker:precheck")
        if terminated:
            return terminated

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
            _, terminated = self._check_budget(state, node="worker:model_call")
            if terminated:
                return terminated
            with span("graph.worker"):
                result = await self.worker.execute(payload)

        updated: AgentGraphState = {
            **state,
            "worker_result": result.model_dump(),
            "evidence": [item.model_dump() for item in evidence],
            "iterations": state.get("iterations", 0) + 1,
        }
        self._log_node(
            logging.INFO,
            "graph_node_complete",
            node="worker",
            state=updated,
            answer_chars=len(result.answer),
            claims=len(result.claims),
            confidence=result.confidence,
        )
        return updated

    async def _verify(self, state: AgentGraphState) -> AgentGraphState:
        budget = self._budget(state)
        budget.verifier_rounds += 1
        self._log_node(logging.INFO, "graph_node_start", node="verify", state=state)

        try:
            budget.check()
        except BudgetExceeded as exc:
            report = VerificationReport(
                verdict="revise",
                task_complete=False,
                issues=[
                    VerificationIssue(
                        code="budget_exhausted",
                        description=str(exc),
                        severity="high",
                    )
                ],
                required_actions=[],
                confidence=0.0,
            )
            return self._budget_termination(
                state,
                node="verify:round_limit",
                exc=exc,
                verification=report,
            )

        issues = deterministic_output_issues(state["worker_result"]["answer"])
        if self.settings.llm_backend != "ollama":
            report = VerificationReport(
                verdict="pass" if not issues else "revise",
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
                    verdict="revise",
                    task_complete=False,
                    issues=[
                        VerificationIssue(
                            code="budget_exhausted",
                            description=str(exc),
                            severity="high",
                        )
                    ],
                    required_actions=[],
                    confidence=0.0,
                )
                return self._budget_termination(
                    state,
                    node="verify:model_call",
                    exc=exc,
                    verification=report,
                )
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

        updated: AgentGraphState = {**state, "verification": report.model_dump()}
        self._log_node(
            logging.INFO,
            "graph_verification_result",
            node="verify",
            state=updated,
            verdict=report.verdict,
            task_complete=report.task_complete,
            issue_count=len(report.issues),
            required_action_count=len(report.required_actions),
            confidence=report.confidence,
        )
        return updated

    async def _revise(self, state: AgentGraphState) -> AgentGraphState:
        self._log_node(logging.INFO, "graph_node_start", node="revise", state=state)
        budget = self._budget(state)
        budget.model_calls += 1
        _, terminated = self._check_budget(state, node="revise:model_call")
        if terminated:
            return terminated

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
        updated: AgentGraphState = {
            **state,
            "worker_result": result.model_dump(),
            "iterations": state.get("iterations", 0) + 1,
        }
        self._log_node(
            logging.INFO,
            "graph_node_complete",
            node="revise",
            state=updated,
            answer_chars=len(result.answer),
            claims=len(result.claims),
            confidence=result.confidence,
        )
        return updated

    async def _research(self, state: AgentGraphState) -> AgentGraphState:
        rounds = state.get("research_rounds", 0) + 1
        self._log_node(
            logging.INFO,
            "graph_node_start",
            node="research",
            state=state,
            requested_round=rounds,
        )

        if rounds > self.settings.phase2_max_research_rounds or not self.settings.mcp_enabled:
            verification = dict(state["verification"])
            verification["verdict"] = "revise"
            updated: AgentGraphState = {
                **state,
                "verification": verification,
                "research_rounds": rounds,
            }
            self._log_node(
                logging.WARNING,
                "graph_research_skipped",
                node="research",
                state=updated,
                reason=(
                    "maximum research rounds exceeded"
                    if rounds > self.settings.phase2_max_research_rounds
                    else "mcp disabled"
                ),
            )
            return updated

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
            log_kv(
                logger,
                logging.INFO,
                "graph_research_result",
                tool=tool,
                round=rounds,
                ok=result.ok,
                error=result.error,
            )
        except ToolApprovalRequired as exc:
            metadata.setdefault("evidence", []).append(
                {"id": f"research-{rounds}", "source": tool, "content": str(exc)}
            )
            log_kv(
                logger,
                logging.WARNING,
                "graph_research_approval_required",
                tool=tool,
                round=rounds,
                error=str(exc),
            )
        except BudgetExceeded as exc:
            return self._budget_termination(state, node="research:tool_call", exc=exc)

        updated = {**state, "metadata": metadata, "research_rounds": rounds}
        self._log_node(
            logging.INFO,
            "graph_node_complete",
            node="research",
            state=updated,
            tool=tool,
        )
        return updated

    async def _replan(self, state: AgentGraphState) -> AgentGraphState:
        replans = state.get("replans", 0) + 1
        self._log_node(
            logging.INFO,
            "graph_node_start",
            node="replan",
            state=state,
            requested_replan=replans,
        )
        if replans > self.settings.phase2_max_replans:
            verification = dict(state["verification"])
            verification["verdict"] = "revise"
            updated: AgentGraphState = {
                **state,
                "verification": verification,
                "replans": replans,
            }
            self._log_node(
                logging.WARNING,
                "graph_replan_limit_reached",
                node="replan",
                state=updated,
            )
            return updated

        budget = self._budget(state)
        budget.model_calls += 1
        _, terminated = self._check_budget(state, node="replan:model_call")
        if terminated:
            return terminated

        plan = (
            await self.planner.plan(state["message"])
            if self.settings.llm_backend == "ollama"
            else ExecutionPlan.model_validate(state["plan"])
        )
        updated = {
            **state,
            "plan": plan.model_dump(),
            "task_index": 0,
            "replans": replans,
        }
        self._log_node(
            logging.INFO,
            "graph_node_complete",
            node="replan",
            state=updated,
            planned_tasks=len(plan.tasks),
        )
        return updated

    async def _advance(self, state: AgentGraphState) -> AgentGraphState:
        results = list(state.get("task_results", []))
        results.append(
            {
                "task": self._current_task(state),
                "worker_result": state["worker_result"],
                "verification": state["verification"],
            }
        )
        updated: AgentGraphState = {
            **state,
            "task_results": results,
            "task_index": state.get("task_index", 0) + 1,
            "verification": {},
            "worker_result": {},
        }
        self._log_node(
            logging.INFO,
            "graph_task_advanced",
            node="advance",
            state=updated,
            completed_tasks=len(results),
        )
        return updated

    async def _finalize(self, state: AgentGraphState) -> AgentGraphState:
        self._log_node(logging.INFO, "graph_node_start", node="finalize", state=state)
        results = state.get("task_results", [])
        if state.get("termination_reason"):
            return await self._terminate(state)

        if self.settings.llm_backend == "ollama" and len(results) > 1:
            budget = self._budget(state)
            budget.model_calls += 1
            _, terminated = self._check_budget(state, node="finalize:model_call")
            if terminated:
                return await self._terminate(terminated)
            response = await self.synthesizer.synthesize(
                {"user_request": state["message"], "verified_results": results}
            )
        else:
            response = (
                results[-1]["worker_result"]["answer"]
                if results
                else state.get("worker_result", {}).get("answer", "")
            )

        updated: AgentGraphState = {
            **state,
            "response": response,
            "backend": self.settings.llm_backend,
            "model": self.settings.model_general,
        }
        self._log_node(
            logging.INFO,
            "graph_node_complete",
            node="finalize",
            state=updated,
            response_chars=len(response),
            completed_tasks=len(results),
        )
        return updated

    async def _terminate(self, state: AgentGraphState) -> AgentGraphState:
        response = self._partial_response(state)
        metrics.inc("chat.partial_response")
        updated: AgentGraphState = {
            **state,
            "response": response,
            "backend": self.settings.llm_backend,
            "model": self.settings.model_general,
        }
        self._log_node(
            logging.WARNING,
            "graph_execution_terminated",
            node="terminate",
            state=updated,
            reason=state.get("termination_reason"),
            response_chars=len(response),
            completed_tasks=len(state.get("task_results", [])),
            current_task_has_output=bool(
                (state.get("worker_result") or {}).get("answer")
            ),
        )
        return updated

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
        log_kv(
            logger,
            logging.INFO,
            "graph_request_start",
            thread_id=thread_id,
            run_id=request.metadata.get("run_id"),
            backend=self.settings.llm_backend,
            planner_model=self.settings.model_planner,
            worker_model=self.settings.model_general,
            verifier_model=self.settings.model_reasoning,
            max_duration_seconds=budget.max_duration_seconds,
            max_model_calls=budget.max_model_calls,
            max_tool_calls=budget.max_tool_calls,
            max_verifier_rounds=budget.max_verifier_rounds,
            history_messages=len(history),
        )

        try:
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
        except BudgetExceeded as exc:
            # Final containment guard. Normal budget exhaustion is handled by
            # graph nodes and routed to `terminate`; this prevents any missed
            # path from surfacing as an HTTP 500.
            metrics.inc("graph.unhandled_budget_exhausted")
            log_kv(
                logger,
                logging.ERROR,
                "graph_unhandled_budget_exhausted",
                thread_id=thread_id,
                run_id=request.metadata.get("run_id"),
                reason=str(exc),
                **self._budget_fields(budget),
            )
            result = {
                "response": (
                    "Execution stopped safely before a verified answer could be "
                    f"produced: {exc}."
                ),
                "backend": self.settings.llm_backend,
                "model": self.settings.model_general,
                "termination_reason": str(exc),
                "plan": None,
                "task_results": [],
                "iterations": 0,
            }

        await self.memory.append(
            thread_id,
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": result["response"]},
        )
        log_kv(
            logger,
            logging.INFO,
            "graph_request_complete",
            thread_id=thread_id,
            run_id=request.metadata.get("run_id"),
            termination_reason=result.get("termination_reason"),
            response_chars=len(result["response"]),
            **self._budget_fields(budget),
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
