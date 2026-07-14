from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.agents.base import StructuredOutputError
from app.agents.final_verifier import FinalVerifierAgent
from app.agents.planner import PlannerAgent
from app.agents.prompts import PLANNER_PROMPT
from app.agents.synthesizer import SynthesizerAgent
from app.agents.verifier import VerifierAgent
from app.agents.worker import WorkerAgent
from app.graphs.routes import (
    after_advance,
    after_budgeted_step,
    after_plan,
    after_verification,
)
from app.graphs.state import AgentGraphState
from app.llm.ollama import OllamaClient
from app.logging_config import log_kv
from app.mcp.client import MCPClient
from app.observability.events import event
from app.observability.metrics import metrics
from app.observability.tracing import span
from app.orchestration.execution_meter import (
    execution_meter_scope,
    get_current_execution_meter,
    model_operation_scope,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.evidence import EvidenceItem
from app.schemas.execution import BudgetExceeded, ExecutionBudget
from app.schemas.planning import ExecutionPlan, PlanTask
from app.schemas.verification import VerificationIssue, VerificationReport
from app.schemas.worker import WorkerResult
from app.services.answer_quality import deterministic_output_issues
from app.services.claim_grounding import ground_claims
from app.services.context_builder import build_context, context_character_count
from app.services.evidence import (
    deduplicate_evidence,
    evidence_from_metadata,
    evidence_from_tool_result,
)
from app.services.inventory import (
    InventoryService,
    RuntimeInventory,
    normalize_inventory,
)
from app.services.routing import (
    RuntimeRouter,
    TaskRoutingDecision,
    build_fresh_run_state,
)
from app.settings import Settings
from app.state import StateRuntime
from app.tools.executor import (
    ToolApprovalRequired,
    ToolExecutionDenied,
    ToolExecutor,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


def encode_sse(event_name: str, data: object) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatAgent:
    """Bounded LangGraph runtime using shared model/tool gateways."""

    def __init__(
        self,
        settings: Settings,
        *,
        ollama_client: OllamaClient | None = None,
        mcp_client: MCPClient | None = None,
    ) -> None:
        self.settings = settings
        self.ollama = ollama_client or OllamaClient(settings)
        self.mcp = mcp_client or MCPClient(settings)
        self.planner = self._new_role_agent(PlannerAgent, settings.model_planner)
        self.worker = self._new_role_agent(WorkerAgent, settings.model_general)
        self.verifier = self._new_role_agent(VerifierAgent, settings.model_reasoning)
        self.synthesizer = self._new_role_agent(
            SynthesizerAgent, settings.model_synthesis
        )
        self.tool_executor = ToolExecutor(self.mcp, settings)
        self.router = RuntimeRouter(settings)
        self.selector = self.router.models
        self.inventory_service = InventoryService(settings, self.ollama, self.mcp)
        self.state_runtime = StateRuntime(settings)
        self.memory = self.state_runtime.conversations
        self.graph = self._build_graph(self.state_runtime.checkpointer)
        self.startup_dependencies: dict[str, dict[str, object]] = {}

    def _new_role_agent(self, agent_type: type[T], model: str) -> T:
        agent = agent_type(self.settings, model)
        if hasattr(agent, "ollama"):
            setattr(agent, "ollama", self.ollama)
        return agent

    async def start(self) -> None:
        self.startup_dependencies = {}
        try:
            await self.state_runtime.start()
            self.startup_dependencies["persistence"] = {"status": "available"}
        except Exception as exc:
            required = bool(
                self.settings.persistence_required
                or (
                    self.settings.artifact_backend == "minio"
                    and self.settings.artifact_storage_required
                )
            )
            self.startup_dependencies["persistence"] = {
                "status": "unavailable",
                "required": required,
                "error": f"{type(exc).__name__}: {str(exc).strip()}",
            }
            if required:
                raise
            await self.state_runtime.use_memory_fallback(str(exc))
            self.startup_dependencies["persistence"]["status"] = "degraded"

        self.memory = self.state_runtime.conversations
        self.graph = self._build_graph(self.state_runtime.checkpointer)

        if self.settings.llm_backend == "ollama":
            try:
                start_ollama = getattr(self.ollama, "start", None)
                if callable(start_ollama):
                    await start_ollama()
                self.startup_dependencies["ollama"] = {"status": "available"}
            except Exception as exc:
                self.startup_dependencies["ollama"] = {
                    "status": "unavailable",
                    "required": self.settings.ollama_required,
                    "error": f"{type(exc).__name__}: {str(exc).strip()}",
                }
                if self.settings.ollama_required:
                    raise

        if self.settings.mcp_enabled:
            try:
                start_mcp = getattr(self.mcp, "start", None)
                if callable(start_mcp):
                    await start_mcp()
                self.startup_dependencies["mcp"] = {"status": "available"}
            except Exception as exc:
                self.startup_dependencies["mcp"] = {
                    "status": "unavailable",
                    "required": self.settings.mcp_required,
                    "error": f"{type(exc).__name__}: {str(exc).strip()}",
                }
                if self.settings.mcp_required:
                    raise

    def dependency_startup_status(self) -> dict[str, dict[str, object]]:
        return {key: dict(value) for key, value in self.startup_dependencies.items()}

    async def aclose(self) -> None:
        closers: list[Awaitable[object]] = []
        for dependency in (self.ollama, self.mcp, self.state_runtime):
            close = getattr(dependency, "aclose", None)
            if callable(close):
                closers.append(close())
        if closers:
            await asyncio.gather(*closers, return_exceptions=True)

    async def persistence_health(self) -> dict[str, Any]:
        return await self.state_runtime.health()

    async def load_inventory(self) -> dict[str, object]:
        return (await self.inventory_service.load()).as_dict()

    def _build_graph(self, checkpointer: Any):
        builder = StateGraph(AgentGraphState)
        builder.add_node("plan", self._plan)
        builder.add_node("research", self._research)
        builder.add_node("worker", self._worker)
        builder.add_node("verify", self._verify)
        builder.add_node("revise", self._revise)
        builder.add_node("replan", self._replan)
        builder.add_node("advance", self._advance)
        builder.add_node("finalize", self._finalize)
        builder.add_node("verify_final", self._verify_final)
        builder.add_node("revise_final", self._revise_final)
        builder.add_node("terminate", self._terminate)

        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan",
            after_plan,
            {"research": "research", "worker": "worker", "terminate": "terminate"},
        )
        for node, next_node in (
            ("research", "worker"),
            ("worker", "verify"),
            ("revise", "verify"),
            ("replan", "worker"),
        ):
            builder.add_conditional_edges(
                node,
                after_budgeted_step,
                {"continue": next_node, "terminate": "terminate"},
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
            "advance",
            after_advance,
            {"research": "research", "worker": "worker", "finalize": "finalize"},
        )
        builder.add_conditional_edges(
            "finalize",
            self._after_finalize,
            {"complete": END, "verify_final": "verify_final", "terminate": "terminate"},
        )
        builder.add_conditional_edges(
            "verify_final",
            self._after_final_verification,
            {"complete": END, "revise_final": "revise_final", "terminate": "terminate"},
        )
        builder.add_edge("revise_final", "verify_final")
        builder.add_edge("terminate", END)
        return builder.compile(checkpointer=checkpointer)

    @staticmethod
    def _budget(state: AgentGraphState) -> ExecutionBudget:
        current = get_current_execution_meter()
        if current is not None:
            return current
        value = state.get("execution_budget")  # compatibility for direct node tests
        if isinstance(value, ExecutionBudget):
            return value
        snapshot = state.get("execution_meter_state")
        if isinstance(snapshot, dict):
            raise RuntimeError(
                "serialized execution meter requires a request-scoped runtime meter"
            )
        raise RuntimeError("request-scoped execution meter is missing")

    @staticmethod
    def _checkpoint_safe_state(
        state: AgentGraphState, budget: ExecutionBudget
    ) -> AgentGraphState:
        """Return graph input containing data-only execution-meter state."""

        clean = {
            key: value for key, value in state.items() if key != "execution_budget"
        }
        clean["execution_meter_state"] = budget.snapshot().model_dump(mode="json")
        return clean

    @classmethod
    def _state_update(
        cls, state: AgentGraphState, **updates: object
    ) -> AgentGraphState:
        budget = cls._budget(state)
        clean = {
            key: value for key, value in state.items() if key != "execution_budget"
        }
        # Direct node unit tests do not enter the request runtime context. Preserve
        # their in-memory budget only for that non-checkpointed path; graph runs
        # always have a request-scoped meter and persist the JSON snapshot only.
        if get_current_execution_meter() is None and isinstance(
            state.get("execution_budget"), ExecutionBudget
        ):
            clean["execution_budget"] = state["execution_budget"]
        return {
            **clean,
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
        task_id = None
        if 0 <= index < len(tasks) and isinstance(tasks[index], dict):
            task_id = tasks[index].get("id")
        metadata = state.get("metadata") or {}
        return {
            "run_id": state.get("run_id") or metadata.get("run_id"),
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
        try:
            budget_fields = self._budget_fields(self._budget(state))
        except RuntimeError:
            budget_fields = {}
        log_kv(
            logger,
            level,
            event_name,
            node=node,
            **self._task_fields(state),
            **budget_fields,
            **fields,
        )

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

    def _budget_termination(
        self,
        state: AgentGraphState,
        *,
        node: str,
        exc: BudgetExceeded,
        verification: VerificationReport | None = None,
    ) -> AgentGraphState:
        update: dict[str, object] = {"termination_reason": str(exc)}
        if verification is not None:
            update["verification"] = verification.model_dump()
        metrics.inc("graph.budget_exhausted")
        return self._state_update(state, **update)

    def _check_budget(
        self,
        state: AgentGraphState,
        *,
        node: str,
    ) -> tuple[ExecutionBudget, AgentGraphState | None]:
        budget = self._budget(state)
        try:
            budget.check()
        except BudgetExceeded as exc:
            return budget, self._budget_termination(state, node=node, exc=exc)
        return budget, None

    @staticmethod
    def _state_evidence(state: AgentGraphState) -> list[EvidenceItem]:
        output: list[EvidenceItem] = []
        for raw in state.get("evidence", []):
            try:
                output.append(EvidenceItem.model_validate(raw))
            except Exception:
                continue
        return output

    def _inventory(self, state: AgentGraphState) -> RuntimeInventory:
        return normalize_inventory(state.get("inventory") or {})

    def _current_task(self, state: AgentGraphState) -> dict[str, Any]:
        return state["plan"]["tasks"][state.get("task_index", 0)]

    def _route_current_task(self, state: AgentGraphState) -> TaskRoutingDecision:
        return self.router.classify_task(
            user_request=state["message"],
            task=self._current_task(state),
        )

    def _select_model(
        self,
        state: AgentGraphState,
        *,
        node: str,
        role: str,
        reason: str,
    ) -> str:
        decision = self.router.select_model(
            role=role,
            inventory=self._inventory(state),
            reason=reason,
        )
        log_kv(
            logger,
            logging.INFO,
            "graph_model_selected",
            node=node,
            role=decision.role,
            model=decision.model,
            reason=decision.reason,
        )
        return decision.model

    @staticmethod
    def _with_selected_model(
        state: AgentGraphState,
        *,
        node: str,
        model: str,
    ) -> dict[str, str]:
        selected = dict(state.get("selected_models") or {})
        selected[node] = model
        plan = state.get("plan") or {}
        tasks = plan.get("tasks") or []
        index = int(state.get("task_index", 0))
        if 0 <= index < len(tasks) and isinstance(tasks[index], dict):
            task_id = str(tasks[index].get("id") or "")
            if task_id:
                selected[f"{node}:{task_id}"] = model
        return selected

    def _set_next_action(self, state: AgentGraphState) -> AgentGraphState:
        decision = self._route_current_task(state)
        task_id = str(self._current_task(state).get("id") or "")
        researched = set(state.get("researched_task_ids") or [])
        action = (
            "research"
            if decision.requires_external_evidence and task_id not in researched
            else "worker"
        )
        routing = {
            "requires_external_evidence": decision.requires_external_evidence,
            "worker_role": decision.worker_role,
            "verifier_role": decision.verifier_role,
            "reason": decision.reason,
            "next_action": action,
        }
        updated = self._state_update(state, next_action=action, routing=routing)
        self._log_node(
            logging.INFO,
            "graph_task_route",
            node="route_task",
            state=updated,
            next_action=action,
            reason=decision.reason,
        )
        return updated

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
        candidate = str(state.get("response") or "").strip()
        if candidate:
            sections.append(
                f"Candidate final answer (not fully verified):\n{candidate}"
            )
        current = state.get("worker_result") or {}
        current_answer = str(current.get("answer") or "").strip()
        if current_answer and all(
            current_answer not in section for section in sections
        ):
            sections.append(
                f"In-progress task output (not fully verified):\n{current_answer}"
            )
        if sections:
            return (
                "\n\n".join(sections)
                + "\n\nExecution stopped safely before all work was completed: "
                + reason
                + ". Treat any in-progress output as unverified."
            )
        return (
            "Execution stopped safely before a verified answer could be produced: "
            f"{reason}."
        )

    async def _plan(self, state: AgentGraphState) -> AgentGraphState:
        _, terminated = self._check_budget(state, node="plan")
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
            planner_model = "echo"
        else:
            planner_model = self._select_model(
                state,
                node="plan",
                role="planner",
                reason="planning role",
            )
            planner = self._new_role_agent(PlannerAgent, planner_model)
            try:
                with span("graph.plan"):
                    plan = await self._invoke_model(
                        state,
                        lambda: planner.invoke_json(
                            system=PLANNER_PROMPT,
                            payload={
                                "request": state["message"],
                                "runtime_context": self.router.execution_context(),
                            },
                            schema=ExecutionPlan,
                        ),
                    )
            except StructuredOutputError as exc:
                return self._state_update(
                    state,
                    termination_reason=(
                        f"plan could not produce valid structured output after "
                        f"{len(exc.attempted_models)} bounded attempt(s)"
                    ),
                )
        evidence = evidence_from_metadata(
            state.get("metadata", {}),
            run_id=str(state.get("run_id") or "request"),
            task_id="request",
        )
        updated = self._state_update(
            state,
            plan=plan.model_dump(),
            task_index=0,
            task_results=[],
            worker_result={},
            verification={},
            evidence=[item.model_dump(mode="json") for item in evidence],
            iterations=0,
            research_rounds=0,
            replans=0,
            termination_reason=None,
            response="",
            selected_models=self._with_selected_model(
                state,
                node="plan",
                model=planner_model,
            ),
            selected_tool=None,
            selected_tools={},
            researched_task_ids=[],
            research_queries={},
        )
        return self._set_next_action(updated)

    async def _worker(self, state: AgentGraphState) -> AgentGraphState:
        _, terminated = self._check_budget(state, node="worker")
        if terminated:
            return terminated
        task = self._current_task(state)
        evidence = self._state_evidence(state)
        context_limit = int(self.settings.agent_max_context_chars)
        context = build_context(
            evidence,
            context_limit,
            max_item_chars=min(
                int(
                    getattr(
                        self.settings, "research_max_evidence_chars_per_query", 6000
                    )
                ),
                context_limit,
            ),
        )
        routing = self._route_current_task(state)
        log_kv(
            logger,
            logging.INFO,
            "graph_worker_context_prepared",
            evidence_items=len(context),
            evidence_chars=context_character_count(context),
            context_limit_chars=context_limit,
        )
        payload = {
            "user_request": state["message"],
            "system_instruction": state.get("system_prompt"),
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
            worker_model = "echo"
        else:
            worker_model = self._select_model(
                state,
                node="worker",
                role=routing.worker_role,
                reason=routing.reason,
            )
            worker = self._new_role_agent(WorkerAgent, worker_model)
            try:
                result = await self._invoke_model(
                    state,
                    lambda: worker.execute(payload),
                )
            except StructuredOutputError as exc:
                return self._state_update(
                    state,
                    termination_reason=(
                        f"worker could not produce valid structured output after "
                        f"{len(exc.attempted_models)} bounded attempt(s)"
                    ),
                )
        return self._state_update(
            state,
            worker_result=result.model_dump(),
            evidence=[item.model_dump(mode="json") for item in evidence],
            iterations=state.get("iterations", 0) + 1,
            selected_models=self._with_selected_model(
                state,
                node="worker",
                model=worker_model,
            ),
            model=worker_model,
        )

    async def _verify(self, state: AgentGraphState) -> AgentGraphState:
        budget = self._budget(state)
        budget.verifier_rounds += 1
        try:
            budget.check()
        except BudgetExceeded as exc:
            return self._budget_termination(
                state,
                node="verify",
                exc=exc,
                verification=self._termination_report(exc),
            )

        worker_result = WorkerResult.model_validate(state["worker_result"])
        evidence = self._state_evidence(state)
        task_id = str(self._current_task(state).get("id") or "task")
        grounding = ground_claims(
            worker_result.claims,
            evidence,
            run_id=str(state.get("run_id") or "request"),
            task_id=task_id,
        )
        deterministic = deterministic_output_issues(worker_result.answer)
        grounding_issues = [
            f"claim_grounding:{item.claim_id}:{item.status}"
            for item in grounding
            if item.status != "supported"
        ]
        issues = [*deterministic, *grounding_issues]
        routing = self._route_current_task(state)
        if self.settings.llm_backend != "ollama":
            report = VerificationReport(
                verdict="pass" if not issues else "revise",
                task_complete=not issues,
                issues=[
                    VerificationIssue(code=item, description=item) for item in issues
                ],
                confidence=0.5,
            )
            verifier_model = "echo"
        else:
            verifier_model = self._select_model(
                state,
                node="verify",
                role=routing.verifier_role,
                reason="independent verification",
            )
            verifier = self._new_role_agent(VerifierAgent, verifier_model)
            try:
                report = await self._invoke_model(
                    state,
                    lambda: verifier.verify(
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
                return self._budget_termination(
                    state,
                    node="verify",
                    exc=exc,
                    verification=self._termination_report(exc),
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
            selected_models=self._with_selected_model(
                state,
                node="verify",
                model=verifier_model,
            ),
        )

    async def _revise(self, state: AgentGraphState) -> AgentGraphState:
        budget = self._budget(state)
        budget.state.revision_rounds += 1
        _, terminated = self._check_budget(state, node="revise")
        if terminated:
            return terminated
        routing = self._route_current_task(state)
        model = (
            self._select_model(
                state,
                node="revise",
                role=routing.worker_role,
                reason="revision role",
            )
            if self.settings.llm_backend == "ollama"
            else "echo"
        )
        payload = {
            "user_request": state["message"],
            "task": self._current_task(state),
            "worker_result": state["worker_result"],
            "verification": state["verification"],
            "evidence": state.get("evidence", []),
        }
        if self.settings.llm_backend == "ollama":
            worker = self._new_role_agent(WorkerAgent, model)
            result = await self._invoke_model(state, lambda: worker.revise(payload))
        else:
            result = WorkerResult.model_validate(state["worker_result"])
        return self._state_update(
            state,
            worker_result=result.model_dump(),
            iterations=state.get("iterations", 0) + 1,
            selected_models=self._with_selected_model(
                state,
                node="revise",
                model=model,
            ),
            model=model,
        )

    async def _research(self, state: AgentGraphState) -> AgentGraphState:
        rounds = state.get("research_rounds", 0) + 1
        budget = self._budget(state)
        budget.state.research_rounds = rounds
        if not self.settings.mcp_enabled:
            return self._state_update(
                state,
                research_rounds=rounds,
                termination_reason="external evidence is required but MCP is disabled",
            )
        if rounds > self.settings.agent_max_research_rounds:
            return self._budget_termination(
                state,
                node="research",
                exc=BudgetExceeded("maximum research rounds exceeded"),
            )

        metadata = dict(state.get("metadata", {}))
        verification = state.get("verification") or {}
        required_actions = verification.get("required_actions") or []
        task = self._current_task(state)
        task_id = str(task.get("id") or f"task-{state.get('task_index', 0)}")
        queries = self.router.build_research_queries(
            user_request=state["message"],
            task=task,
            required_actions=required_actions,
            limit=self.settings.research_max_queries_per_task,
        )
        research_queries = dict(state.get("research_queries") or {})
        research_queries[task_id] = queries
        evidence = self._state_evidence(state)
        failures: list[dict[str, str]] = []
        successful_tools: list[str] = []

        for query_index, query in enumerate(queries, start=1):
            candidates = self.router.select_tools(
                user_request=state["message"],
                task=task,
                required_actions=required_actions,
                inventory=self._inventory(state),
                metadata=metadata,
                limit=3,
                query=query,
            )
            log_kv(
                logger,
                logging.INFO,
                "graph_tool_candidates",
                task_id=task_id,
                candidates=",".join(item.name for item in candidates),
            )
            query_succeeded = False
            for candidate in candidates:
                log_kv(
                    logger,
                    logging.INFO,
                    "graph_tool_selected",
                    task_id=task_id,
                    tool=candidate.name,
                    reason=candidate.reason,
                )
                try:
                    result = await self.tool_executor.execute(
                        candidate.name,
                        candidate.arguments,
                        budget=budget,
                        metadata=metadata,
                    )
                except (ToolApprovalRequired, ToolExecutionDenied) as exc:
                    failures.append({"tool": candidate.name, "error": str(exc)})
                    continue
                except BudgetExceeded as exc:
                    return self._budget_termination(
                        state,
                        node="research",
                        exc=exc,
                    )
                record = evidence_from_tool_result(
                    result=result,
                    evidence_id=f"research-{task_id}-{rounds}-{query_index}",
                    run_id=str(state.get("run_id") or "request"),
                    task_id=task_id,
                    query_id=self.router.query_fingerprint(query),
                    tool_name=candidate.name,
                    query=query,
                )
                if record.eligible_for_claim_support:
                    evidence.append(record)
                    successful_tools.append(candidate.name)
                    query_succeeded = True
                    break
                failures.append(
                    {
                        "tool": candidate.name,
                        "error": str(result.error or "tool failed"),
                    }
                )
            if not query_succeeded:
                continue

        metadata.setdefault("research_failures", []).extend(failures)
        if not successful_tools:
            return self._state_update(
                state,
                metadata=metadata,
                research_rounds=rounds,
                research_queries=research_queries,
                evidence=[item.model_dump(mode="json") for item in evidence],
                termination_reason=(
                    "all compatible read-only MCP research attempts failed; "
                    "no current evidence was retrieved"
                ),
            )
        researched = list(
            dict.fromkeys([*(state.get("researched_task_ids") or []), task_id])
        )
        unique_tools = list(dict.fromkeys(successful_tools))
        selected_tools = dict(state.get("selected_tools") or {})
        selected_tools[task_id] = (
            unique_tools[0] if len(unique_tools) == 1 else unique_tools
        )
        return self._state_update(
            state,
            metadata=metadata,
            research_rounds=rounds,
            research_queries=research_queries,
            researched_task_ids=researched,
            selected_tool=unique_tools[0],
            selected_tools=selected_tools,
            evidence=[
                item.model_dump(mode="json") for item in deduplicate_evidence(evidence)
            ],
            next_action="worker",
        )

    async def _replan(self, state: AgentGraphState) -> AgentGraphState:
        replans = state.get("replans", 0) + 1
        budget = self._budget(state)
        budget.state.replans = replans
        if replans > self.settings.agent_max_replans:
            return self._state_update(
                state,
                replans=replans,
                termination_reason="maximum replans exceeded",
            )
        if self.settings.llm_backend == "ollama":
            model = self._select_model(
                state,
                node="replan",
                role="planner",
                reason="replanning role",
            )
            planner = self._new_role_agent(PlannerAgent, model)
            plan = await self._invoke_model(
                state,
                lambda: planner.invoke_json(
                    system=PLANNER_PROMPT,
                    payload={"request": state["message"]},
                    schema=ExecutionPlan,
                ),
            )
        else:
            model = "echo"
            plan = ExecutionPlan.model_validate(state["plan"])
        updated = self._state_update(
            state,
            plan=plan.model_dump(),
            task_index=0,
            task_results=[],
            worker_result={},
            verification={},
            evidence=[],
            research_rounds=0,
            replans=replans,
            researched_task_ids=[],
            research_queries={},
            selected_models=self._with_selected_model(
                state,
                node="replan",
                model=model,
            ),
        )
        return self._set_next_action(updated)

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
        updated = self._state_update(
            state,
            task_results=results,
            task_index=state.get("task_index", 0) + 1,
            verification={},
            worker_result={},
            grounding=[],
            evidence=[],
            research_rounds=0,
            selected_tool=None,
            next_action="worker",
        )
        tasks = (updated.get("plan") or {}).get("tasks", [])
        if updated.get("task_index", 0) < len(tasks):
            updated = self._set_next_action(updated)
        return updated

    @staticmethod
    def _after_finalize(state: AgentGraphState) -> str:
        if state.get("termination_reason"):
            return "terminate"
        return (
            "verify_final" if state.get("final_verification_required") else "complete"
        )

    def _after_final_verification(self, state: AgentGraphState) -> str:
        if state.get("termination_reason"):
            return "terminate"
        report = state.get("final_verification") or {}
        if report.get("verdict") == "pass" and report.get("answer_complete") is True:
            return "complete"
        if (
            int(state.get("final_revision_rounds", 0))
            < self.settings.final_max_revision_rounds
        ):
            return "revise_final"
        return "terminate"

    async def _finalize(self, state: AgentGraphState) -> AgentGraphState:
        if state.get("termination_reason"):
            return await self._terminate(state)
        results = state.get("task_results", [])
        model = str(state.get("model") or self.settings.model_general)
        verification_required = False
        if self.settings.llm_backend == "ollama" and len(results) > 1:
            model = self._select_model(
                state,
                node="finalize",
                role="synthesis",
                reason="multi-task synthesis",
            )
            synthesizer = self._new_role_agent(SynthesizerAgent, model)
            try:
                response = await self._invoke_model(
                    state,
                    lambda: synthesizer.synthesize(
                        {
                            "user_request": state["message"],
                            "verified_results": results,
                        }
                    ),
                )
            except Exception as exc:
                return await self._terminate(
                    self._state_update(
                        state,
                        termination_reason=f"final synthesis failed: {type(exc).__name__}",
                    )
                )
            verification_required = bool(self.settings.final_verification_enabled)
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
            model=model,
            final_verification_required=verification_required,
            final_verification={},
            final_revision_rounds=0,
            selected_models=self._with_selected_model(
                state,
                node="finalize",
                model=model,
            ),
        )

    async def _verify_final(self, state: AgentGraphState) -> AgentGraphState:
        _, terminated = self._check_budget(state, node="verify_final")
        if terminated:
            return terminated
        model = self._select_model(
            state,
            node="verify_final",
            role="reasoning",
            reason="independent verification of synthesized final answer",
        )
        verifier = self._new_role_agent(FinalVerifierAgent, model)
        try:
            report = await self._invoke_model(
                state,
                lambda: verifier.verify_final(
                    {
                        "user_request": state["message"],
                        "verified_results": state.get("task_results", []),
                        "candidate_final_answer": state.get("response", ""),
                    }
                ),
            )
        except StructuredOutputError as exc:
            return self._state_update(
                state,
                termination_reason=(
                    f"final verification could not produce valid structured output "
                    f"after {len(exc.attempted_models)} bounded attempt(s)"
                ),
            )
        updated = self._state_update(
            state,
            final_verification=report.model_dump(),
            selected_models=self._with_selected_model(
                state,
                node="verify_final",
                model=model,
            ),
        )
        if (
            report.verdict != "pass"
            and int(state.get("final_revision_rounds", 0))
            >= self.settings.final_max_revision_rounds
        ):
            updated["termination_reason"] = (
                "final answer could not be independently verified"
            )
        return updated

    async def _revise_final(self, state: AgentGraphState) -> AgentGraphState:
        rounds = int(state.get("final_revision_rounds", 0)) + 1
        budget = self._budget(state)
        budget.state.final_revision_rounds = rounds
        _, terminated = self._check_budget(state, node="revise_final")
        if terminated:
            return terminated
        model = self._select_model(
            state,
            node="revise_final",
            role="synthesis",
            reason="final answer revision",
        )
        synthesizer = self._new_role_agent(SynthesizerAgent, model)
        try:
            response = await self._invoke_model(
                state,
                lambda: synthesizer.synthesize(
                    {
                        "user_request": state["message"],
                        "verified_results": state.get("task_results", []),
                        "previous_answer": state.get("response", ""),
                        "final_verification": state.get("final_verification", {}),
                        "instruction": (
                            "Revise only as required; do not introduce new facts."
                        ),
                    }
                ),
            )
        except Exception as exc:
            return self._state_update(
                state,
                termination_reason=(
                    f"final answer revision failed: {type(exc).__name__}"
                ),
                final_revision_rounds=rounds,
            )
        return self._state_update(
            state,
            response=response,
            final_revision_rounds=rounds,
            final_verification={},
            selected_models=self._with_selected_model(
                state,
                node="revise_final",
                model=model,
            ),
        )

    async def _terminate(self, state: AgentGraphState) -> AgentGraphState:
        return self._state_update(
            state,
            response=self._partial_response(state),
            backend=self.settings.llm_backend,
            model=self.settings.model_general,
        )

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        conversation_id = request.conversation_id or request.thread_id or str(uuid4())
        run_id = request.run_id or str(uuid4())
        execution_thread_id = f"{conversation_id}:{run_id}"
        request_id = str(uuid4())
        budget = ExecutionBudget(
            self.settings.execution_max_duration_seconds,
            self.settings.execution_max_model_calls,
            self.settings.execution_max_tool_calls,
            self.settings.execution_max_verifier_rounds,
        )
        history = await self.memory.get(conversation_id)
        runtime_inventory = await self.inventory_service.load()
        system_decision = self.router.prepare_system_prompt(
            message=request.message,
            provided=request.system_prompt,
        )
        config = {"configurable": {"thread_id": execution_thread_id}}
        initial_state = build_fresh_run_state(
            message=request.message,
            system_prompt=system_decision.prompt,
            system_prompt_source=system_decision.source,
            request_domain=system_decision.domain,
            metadata={
                **request.metadata,
                "conversation_id": conversation_id,
                "run_id": run_id,
                "execution_thread_id": execution_thread_id,
            },
            history=history,
            execution_budget=budget,
            request_id=request_id,
            inventory=runtime_inventory,
            backend=self.settings.llm_backend,
        )
        initial_state.update(
            {
                "conversation_id": conversation_id,
                "run_id": run_id,
                "execution_thread_id": execution_thread_id,
            }
        )
        initial_state = self._checkpoint_safe_state(initial_state, budget)
        log_kv(
            logger,
            logging.INFO,
            "graph_request_start",
            thread_id=conversation_id,
            conversation_id=conversation_id,
            run_id=run_id,
            execution_thread_id=execution_thread_id,
            request_id=request_id,
            transient_state_reset=True,
        )
        log_kv(
            logger,
            logging.INFO,
            "graph_runtime_inventory",
            model_count=len(runtime_inventory.model_names),
            tool_count=len(runtime_inventory.tool_names),
            cached=runtime_inventory.cached,
        )
        with execution_meter_scope(budget), span("chat.total"):
            try:
                result = await self.graph.ainvoke(initial_state, config=config)
            except BudgetExceeded as exc:
                result = self._state_update(
                    initial_state,
                    response=(
                        "Execution stopped safely before a verified answer could be "
                        f"produced: {exc}."
                    ),
                    backend=self.settings.llm_backend,
                    model=None,
                    termination_reason=str(exc),
                    task_results=[],
                )
        await self.memory.append(
            conversation_id,
            {"role": "user", "content": request.message, "run_id": run_id},
            {
                "role": "assistant",
                "content": result["response"],
                "run_id": run_id,
            },
        )
        return ChatResponse.from_result(
            conversation_id=conversation_id,
            run_id=run_id,
            execution_thread_id=execution_thread_id,
            response=result["response"],
            backend=result["backend"],
            model=result.get("model"),
            metadata={
                **request.metadata,
                "runtime_contract": "agent-graph-v1",
                "plan": result.get("plan"),
                "verification": result.get("task_results"),
                "final_verification": result.get("final_verification"),
                "final_revision_rounds": result.get("final_revision_rounds", 0),
                "iterations": result.get("iterations"),
                "termination_reason": result.get("termination_reason"),
                "routing": result.get("routing"),
                "selected_models": result.get("selected_models"),
                "selected_tool": result.get("selected_tool"),
                "selected_tools": result.get("selected_tools"),
                "research_queries": result.get("research_queries"),
                "system_prompt": {
                    "source": system_decision.source,
                    "domain": system_decision.domain,
                    "generated": system_decision.source == "derived",
                },
                "inventory": {
                    "models": runtime_inventory.model_names,
                    "tools": runtime_inventory.tool_names,
                    "errors": runtime_inventory.errors,
                    "cached": runtime_inventory.cached,
                },
                "usage": budget.usage_metadata(),
            },
        )

    async def astream_events(
        self,
        request: ChatRequest,
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
