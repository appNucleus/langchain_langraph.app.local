from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

from langgraph.graph import END, START, StateGraph

from app.agents.planner import PlannerAgent
from app.agents.synthesizer import SynthesizerAgent
from app.agents.verifier import VerifierAgent
from app.agents.worker import WorkerAgent
from app.graphs.routes import after_advance, after_plan, after_verification
from app.graphs.state import AgentGraphState
from app.llm.ollama import OllamaClient
from app.mcp.client import MCPClient
from app.observability.events import event
from app.observability.metrics import metrics
from app.observability.tracing import span
from app.orchestration.execution_meter import (
    execution_meter_scope,
    get_current_execution_meter,
    model_operation_scope,
)
from app.orchestration.run_context import RunIdentity
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.execution import BudgetExceeded, ExecutionBudget
from app.schemas.planning import ExecutionPlan, PlanTask
from app.schemas.verification import VerificationIssue, VerificationReport
from app.schemas.worker import WorkerResult
from app.services.answer_quality import deterministic_output_issues
from app.services.claim_grounding import ground_claims
from app.services.context_builder import build_context
from app.services.evidence import (
    deduplicate_evidence,
    evidence_from_metadata,
    evidence_from_tool_result,
)
from app.settings import Settings
from app.state.in_memory import BoundedInMemoryStore
from app.tools.executor import ToolApprovalRequired, ToolExecutor

T = TypeVar("T")
_RESERVED_METADATA_KEYS = {
    "conversation_id",
    "execution_thread_id",
    "run_id",
    "thread_id",
    "trust_class",
}


def encode_sse(event_name: str, data: object) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class _Selector:
    def __init__(self, settings: Settings):
        self.settings = settings


class ChatAgent:
    """Checkpointed runtime with request-scoped execution metering and evidence trust."""

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
        self.memory = BoundedInMemoryStore(
            ttl_seconds=settings.state_ttl_seconds,
            max_sessions=settings.state_max_sessions,
            max_messages=settings.state_max_history_messages,
        )
        self.graph = self._build_graph()

    async def start(self) -> None:
        if self.settings.llm_backend == "ollama":
            await self.ollama.start()
        if self.settings.mcp_enabled:
            await self.mcp.start()

    async def aclose(self) -> None:
        await asyncio.gather(
            self.ollama.aclose(),
            self.mcp.aclose(),
            return_exceptions=True,
        )

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

    def _build_graph(self):
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
            after_plan,
            {
                "research": "research",
                "worker": "worker",
                "terminate": "terminate",
            },
        )
        builder.add_edge("worker", "verify")
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
        builder.add_edge("revise", "verify")
        builder.add_edge("research", "worker")
        builder.add_edge("replan", "worker")
        builder.add_conditional_edges(
            "advance",
            after_advance,
            {
                "research": "research",
                "worker": "worker",
                "finalize": "finalize",
                "terminate": "terminate",
            },
        )
        builder.add_edge("finalize", END)
        builder.add_edge("terminate", END)
        return builder.compile()

    @staticmethod
    def _budget(state: AgentGraphState) -> ExecutionBudget:
        current = get_current_execution_meter()
        if current is not None:
            return current
        # Temporary compatibility for direct node tests or legacy callers. The
        # runtime never writes this object into checkpoint state.
        value = state.get("execution_budget")  # type: ignore[typeddict-item]
        if isinstance(value, ExecutionBudget):
            return value
        raise RuntimeError("request-scoped execution meter is missing")

    @classmethod
    def _state_update(
        cls, state: AgentGraphState, **updates: object
    ) -> AgentGraphState:
        """Return checkpoint-safe state containing only a serialized meter snapshot."""

        clean_state = {
            key: value for key, value in state.items() if key != "execution_budget"
        }
        budget = cls._budget(state)
        return {
            **clean_state,
            **updates,
            "execution_meter_state": budget.snapshot().model_dump(mode="json"),
        }

    async def _invoke_model(
        self,
        state: AgentGraphState,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        budget = self._budget(state)
        with model_operation_scope(budget):
            return await operation()

    @staticmethod
    def _termination_report(exc: BaseException) -> VerificationReport:
        return VerificationReport(
            verdict="terminate",
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
            with span("graph.plan"):
                plan = await self._invoke_model(
                    state, lambda: self.planner.plan(state["message"])
                )
        return self._state_update(
            state,
            plan=plan.model_dump(),
            task_index=0,
            task_results=[],
            iterations=0,
            research_rounds=0,
            replans=0,
        )

    def _current_task(self, state: AgentGraphState) -> dict[str, Any]:
        return state["plan"]["tasks"][state.get("task_index", 0)]

    def _evidence(self, state: AgentGraphState) -> list[Any]:
        task = self._current_task(state)
        metadata_evidence = evidence_from_metadata(
            state.get("metadata", {}),
            run_id=str(state.get("run_id", "request")),
            task_id=str(task.get("id", "task")),
        )
        stored = state.get("evidence", [])
        for item in stored:
            try:
                from app.schemas.evidence import EvidenceItem

                metadata_evidence.append(EvidenceItem.model_validate(item))
            except (TypeError, ValueError):
                continue
        return deduplicate_evidence(metadata_evidence)

    async def _worker(self, state: AgentGraphState) -> AgentGraphState:
        self._budget(state).check()
        task = self._current_task(state)
        evidence = self._evidence(state)
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
            with span("graph.worker"):
                result = await self._invoke_model(
                    state, lambda: self.worker.execute(payload)
                )
        return self._state_update(
            state,
            worker_result=result.model_dump(),
            evidence=[item.model_dump(mode="json") for item in evidence],
            iterations=state.get("iterations", 0) + 1,
        )

    async def _verify(self, state: AgentGraphState) -> AgentGraphState:
        budget = self._budget(state)
        budget.verifier_rounds += 1
        try:
            budget.check()
        except BudgetExceeded as exc:
            return self._state_update(
                state,
                verification=self._termination_report(exc).model_dump(),
                termination_reason=str(exc),
            )

        output_issues = deterministic_output_issues(state["worker_result"]["answer"])
        worker_result = WorkerResult.model_validate(state["worker_result"])
        evidence = self._evidence(state)
        task_id = str(self._current_task(state).get("id", "task"))
        grounding = ground_claims(
            worker_result.claims,
            evidence,
            run_id=str(state.get("run_id", "request")),
            task_id=task_id,
        )
        grounding_issues = [
            f"claim_grounding:{item.claim_id}:{item.status}"
            for item in grounding
            if item.status != "supported"
        ]
        issues = [*output_issues, *grounding_issues]

        if self.settings.llm_backend != "ollama":
            report = VerificationReport(
                verdict="pass" if not issues else "revise",
                task_complete=not issues,
                issues=[
                    VerificationIssue(code=item, description=item) for item in issues
                ],
                confidence=0.5,
            )
        else:
            try:
                with span("graph.verify"):
                    report = await self._invoke_model(
                        state,
                        lambda: self.verifier.verify(
                            {
                                "user_request": state["message"],
                                "task": self._current_task(state),
                                "worker_result": state["worker_result"],
                                "evidence": [
                                    item.model_dump(mode="json") for item in evidence
                                ],
                                "grounding": [item.model_dump() for item in grounding],
                                "deterministic_issues": issues,
                            }
                        ),
                    )
            except BudgetExceeded as exc:
                return self._state_update(
                    state,
                    verification=self._termination_report(exc).model_dump(),
                    termination_reason=str(exc),
                )
        if grounding_issues and report.verdict == "pass":
            report.verdict = "revise"
            report.task_complete = False
            report.issues.extend(
                VerificationIssue(code=item, description=item, severity="high")
                for item in grounding_issues
            )
        return self._state_update(
            state,
            verification=report.model_dump(),
            grounding=[item.model_dump() for item in grounding],
        )

    async def _revise(self, state: AgentGraphState) -> AgentGraphState:
        budget = self._budget(state)
        budget.state.revision_rounds += 1
        payload = {
            "user_request": state["message"],
            "task": self._current_task(state),
            "worker_result": state["worker_result"],
            "verification": state["verification"],
            "evidence": state.get("evidence", []),
        }
        result = (
            await self._invoke_model(state, lambda: self.worker.revise(payload))
            if self.settings.llm_backend == "ollama"
            else WorkerResult.model_validate(state["worker_result"])
        )
        return self._state_update(
            state,
            worker_result=result.model_dump(),
            iterations=state.get("iterations", 0) + 1,
        )

    async def _research(self, state: AgentGraphState) -> AgentGraphState:
        rounds = state.get("research_rounds", 0) + 1
        budget = self._budget(state)
        budget.state.research_rounds = rounds
        if (
            rounds > self.settings.phase2_max_research_rounds
            or not self.settings.mcp_enabled
        ):
            verification = dict(state["verification"])
            verification["verdict"] = "revise"
            return self._state_update(
                state, verification=verification, research_rounds=rounds
            )

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
        existing = self._evidence(state)
        tool_errors = list(state.get("tool_errors", []))
        try:
            result = await self.tool_executor.execute(
                tool, arguments, budget=budget, metadata=metadata
            )
            record = evidence_from_tool_result(
                result=result,
                evidence_id=f"research-{rounds}",
                run_id=str(state.get("run_id", "request")),
                task_id=str(self._current_task(state).get("id", "task")),
                query_id=f"query-{rounds}",
                tool_name=tool,
                query=query,
            )
            if record.eligible_for_claim_support:
                existing.append(record)
            else:
                tool_errors.append(record.model_dump(mode="json"))
        except (ToolApprovalRequired, BudgetExceeded, TimeoutError) as exc:
            tool_errors.append(
                {
                    "tool_name": tool,
                    "query": query,
                    "error": str(exc),
                    "eligible_for_claim_support": False,
                }
            )
            if isinstance(exc, BudgetExceeded):
                return self._state_update(
                    state,
                    verification=self._termination_report(exc).model_dump(),
                    termination_reason=str(exc),
                    tool_errors=tool_errors,
                    research_rounds=rounds,
                )
        return self._state_update(
            state,
            evidence=[
                item.model_dump(mode="json") for item in deduplicate_evidence(existing)
            ],
            tool_errors=tool_errors,
            research_rounds=rounds,
        )

    async def _replan(self, state: AgentGraphState) -> AgentGraphState:
        replans = state.get("replans", 0) + 1
        self._budget(state).state.replans = replans
        if replans > self.settings.phase2_max_replans:
            verification = dict(state["verification"])
            verification["verdict"] = "revise"
            return self._state_update(state, verification=verification, replans=replans)
        plan = (
            await self._invoke_model(state, lambda: self.planner.plan(state["message"]))
            if self.settings.llm_backend == "ollama"
            else ExecutionPlan.model_validate(state["plan"])
        )
        return self._state_update(
            state, plan=plan.model_dump(), task_index=0, replans=replans
        )

    async def _advance(self, state: AgentGraphState) -> AgentGraphState:
        results = list(state.get("task_results", []))
        results.append(
            {
                "task": self._current_task(state),
                "worker_result": state["worker_result"],
                "verification": state["verification"],
                "grounding": state.get("grounding", []),
                "evidence": state.get("evidence", []),
            }
        )
        return self._state_update(
            state,
            task_results=results,
            task_index=state.get("task_index", 0) + 1,
            verification={},
            worker_result={},
            grounding=[],
        )

    async def _finalize(self, state: AgentGraphState) -> AgentGraphState:
        results = state.get("task_results", [])
        if self.settings.llm_backend == "ollama" and len(results) > 1:
            try:
                response = await self._invoke_model(
                    state,
                    lambda: self.synthesizer.synthesize(
                        {"user_request": state["message"], "verified_results": results}
                    ),
                )
            except BudgetExceeded as exc:
                return await self._terminate(
                    self._state_update(state, termination_reason=str(exc))
                )
        else:
            response = (
                results[-1]["worker_result"]["answer"]
                if results
                else state.get("worker_result", {}).get("answer", "")
            )
        return self._state_update(
            state,
            response=response,
            backend=self.settings.llm_backend,
            model=self.settings.model_general,
        )

    async def _terminate(self, state: AgentGraphState) -> AgentGraphState:
        partial = state.get("worker_result", {}).get("answer")
        if not partial:
            results = state.get("task_results", [])
            partial = results[-1]["worker_result"].get("answer") if results else ""
        reason = state.get("termination_reason") or "execution terminated safely"
        response = partial or f"The request could not be completed safely: {reason}"
        return self._state_update(
            state,
            response=response,
            backend=self.settings.llm_backend,
            model=self.settings.model_general,
            termination_reason=reason,
        )

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        identity = RunIdentity.resolve(
            conversation_id=request.conversation_id,
            run_id=request.run_id,
            legacy_thread_id=request.thread_id,
        )
        metadata = {
            key: value
            for key, value in request.metadata.items()
            if key not in _RESERVED_METADATA_KEYS
        }
        budget = ExecutionBudget(
            self.settings.execution_max_duration_seconds,
            self.settings.execution_max_model_calls,
            self.settings.execution_max_tool_calls,
            self.settings.execution_max_verifier_rounds,
        )
        history = await self.memory.get(identity.conversation_id)
        metrics.inc("chat.requests")
        config = {"configurable": {"thread_id": identity.execution_thread_id}}
        initial_state = {
            "message": request.message,
            "system_prompt": request.system_prompt
            or self.settings.default_system_prompt,
            "metadata": metadata,
            "history": history,
            "conversation_id": identity.conversation_id,
            "run_id": identity.run_id,
            "execution_thread_id": identity.execution_thread_id,
            "execution_meter_state": budget.snapshot().model_dump(mode="json"),
        }
        with execution_meter_scope(budget), span("chat.total"):
            try:
                result = await self.graph.ainvoke(initial_state, config=config)
            except BudgetExceeded as exc:
                result = self._state_update(
                    initial_state,
                    response=f"The request could not be completed safely: {exc}",
                    backend=self.settings.llm_backend,
                    model=self.settings.model_general,
                    termination_reason=str(exc),
                    task_results=[],
                )
        await self.memory.append(
            identity.conversation_id,
            {"role": "user", "content": request.message, "run_id": identity.run_id},
            {
                "role": "assistant",
                "content": result["response"],
                "run_id": identity.run_id,
            },
        )
        return ChatResponse.from_result(
            conversation_id=identity.conversation_id,
            run_id=identity.run_id,
            execution_thread_id=identity.execution_thread_id,
            response=result["response"],
            backend=result["backend"],
            model=result.get("model"),
            metadata={
                **metadata,
                "plan": result.get("plan"),
                "verification": result.get("task_results"),
                "grounding": result.get("grounding"),
                "iterations": result.get("iterations"),
                "termination_reason": result.get("termination_reason"),
                "usage": budget.usage_metadata(),
            },
        )

    async def astream_events(
        self, request: ChatRequest
    ) -> AsyncIterator[dict[str, object]]:
        yield event(
            "request_started",
            thread_id=request.conversation_id or request.thread_id,
            run_id=request.run_id,
        )
        yield event("planning_started")
        task = asyncio.create_task(self.ainvoke(request))
        try:
            while not task.done():
                await asyncio.sleep(0.5)
                yield event("working", stage="agent_graph")
            response = await task
            yield event("completed", response=response.model_dump())
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
