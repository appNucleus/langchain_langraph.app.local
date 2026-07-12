from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.agents.planner import PlannerAgent
from app.agents.prompts import PLANNER_PROMPT
from app.agents.synthesizer import SynthesizerAgent
from app.agents.verifier import VerifierAgent
from app.agents.worker import WorkerAgent
from app.graphs.routes import after_advance, after_budgeted_step, after_plan, after_verification
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
from app.services.inventory import InventoryService, RuntimeInventory, normalize_inventory
from app.services.routing import (
    RuntimeRouter,
    TaskRoutingDecision,
    build_fresh_run_state,
)
from app.settings import Settings
from app.state import StateRuntime
from app.tools.executor import ToolApprovalRequired, ToolExecutor

logger = logging.getLogger(__name__)


def encode_sse(event_name: str, data: object) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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
        self.router = RuntimeRouter(settings)
        self.selector = self.router.models
        self.inventory_service = InventoryService(settings, self.ollama, self.mcp)
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
        return (await self.inventory_service.load()).as_dict()

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
            after_plan,
            {"worker": "worker", "research": "research", "terminate": "terminate"},
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
            "advance",
            after_advance,
            {"worker": "worker", "research": "research", "finalize": "finalize"},
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
    def _inventory(state: AgentGraphState) -> RuntimeInventory:
        return normalize_inventory(state.get("inventory") or {})

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
            run_id=(state.get("metadata") or {}).get("run_id"),
            request_id=state.get("request_id"),
            role=decision.role,
            model=decision.model,
            reason=decision.reason,
            available_models=len(self._inventory(state).model_names),
        )
        return decision.model

    @staticmethod
    def _with_selected_model(
        state: AgentGraphState, *, node: str, model: str
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
        if not (state.get("plan") or {}).get("tasks"):
            return {**state, "next_action": "worker", "routing": {}}
        decision = self._route_current_task(state)
        task_id = str(self._current_task(state).get("id") or "")
        researched = set(state.get("researched_task_ids") or [])
        action = (
            "research"
            if decision.requires_external_evidence
            and task_id not in researched
            else "worker"
        )
        routing = {
            "requires_external_evidence": decision.requires_external_evidence,
            "worker_role": decision.worker_role,
            "verifier_role": decision.verifier_role,
            "reason": decision.reason,
            "next_action": action,
        }
        updated: AgentGraphState = {**state, "next_action": action, "routing": routing}
        self._log_node(
            logging.INFO,
            "graph_task_route",
            node="route_task",
            state=updated,
            next_action=action,
            requires_external_evidence=decision.requires_external_evidence,
            worker_role=decision.worker_role,
            verifier_role=decision.verifier_role,
            reason=decision.reason,
        )
        return updated

    @staticmethod
    def _serialize_tool_data(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

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
            planner_model = "echo"
        else:
            planner_model = self._select_model(
                state, node="plan", role="planner", reason="planning role"
            )
            budget.model_calls += 1
            _, terminated = self._check_budget(state, node="plan:model_call")
            if terminated:
                return terminated
            runtime_context = {
                **self.router.execution_context(),
                "available_mcp_tools": self._inventory(state).tool_names,
                "system_instruction": state.get("system_prompt"),
                "history_is_context_only": True,
            }
            planner = PlannerAgent(self.settings, planner_model)
            with span("graph.plan"):
                plan = await planner.invoke_json(
                    system=PLANNER_PROMPT,
                    payload={
                        "request": state["message"],
                        "runtime_context": runtime_context,
                    },
                    schema=ExecutionPlan,
                )

        result: AgentGraphState = {
            **state,
            "plan": plan.model_dump(),
            "task_index": 0,
            "task_results": [],
            "worker_result": {},
            "verification": {},
            "evidence": [],
            "iterations": 0,
            "research_rounds": 0,
            "replans": 0,
            "termination_reason": None,
            "response": "",
            "selected_models": self._with_selected_model(
                state, node="plan", model=planner_model
            ),
            "selected_tool": None,
            "selected_tools": {},
            "researched_task_ids": [],
        }
        result = self._set_next_action(result)
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
        routing = self._route_current_task(state)
        payload = {
            "user_request": state["message"],
            "system_instruction": state.get("system_prompt"),
            "task": task,
            "evidence": context,
            "history": state.get("history", []),
            "history_rules": [
                "The current user_request and current task are authoritative.",
                "Ignore unrelated prior-topic history.",
                "Do not treat conversation history as external evidence.",
            ],
            "execution_context": self.router.execution_context(),
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
            budget.model_calls += 1
            _, terminated = self._check_budget(state, node="worker:model_call")
            if terminated:
                return terminated
            worker = WorkerAgent(self.settings, worker_model)
            with span("graph.worker"):
                result = await worker.execute(payload)

        updated: AgentGraphState = {
            **state,
            "worker_result": result.model_dump(),
            "evidence": [item.model_dump() for item in evidence],
            "iterations": state.get("iterations", 0) + 1,
            "selected_models": self._with_selected_model(
                state, node="worker", model=worker_model
            ),
            "model": worker_model,
        }
        self._log_node(
            logging.INFO,
            "graph_node_complete",
            node="worker",
            state=updated,
            answer_chars=len(result.answer),
            claims=len(result.claims),
            evidence_items=len(context),
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
                state, node="verify:round_limit", exc=exc, verification=report
            )

        issues = deterministic_output_issues(state["worker_result"]["answer"])
        routing = self._route_current_task(state)
        if self.settings.llm_backend != "ollama":
            report = VerificationReport(
                verdict="pass" if not issues else "revise",
                task_complete=not issues,
                issues=[VerificationIssue(code=item, description=item) for item in issues],
                confidence=0.5,
            )
            verifier_model = "echo"
        else:
            verifier_model = self._select_model(
                state,
                node="verify",
                role=routing.verifier_role,
                reason=(
                    "independent verification of evidence-dependent output"
                    if routing.requires_external_evidence
                    else "independent verification"
                ),
            )
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
                    state, node="verify:model_call", exc=exc, verification=report
                )
            verifier = VerifierAgent(self.settings, verifier_model)
            with span("graph.verify"):
                report = await verifier.verify(
                    {
                        "user_request": state["message"],
                        "system_instruction": state.get("system_prompt"),
                        "execution_context": self.router.execution_context(),
                        "task": self._current_task(state),
                        "worker_result": state["worker_result"],
                        "evidence": state.get("evidence", []),
                        "deterministic_issues": issues,
                        "verification_rules": [
                            "Current factual claims require supplied external evidence.",
                            "Request research when evidence is absent or insufficient.",
                            "A pass verdict requires task_complete=true.",
                        ],
                    }
                )

        updated: AgentGraphState = {
            **state,
            "verification": report.model_dump(),
            "selected_models": self._with_selected_model(
                state, node="verify", model=verifier_model
            ),
        }
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
        routing = self._route_current_task(state)
        revision_model = (
            self._select_model(
                state,
                node="revise",
                role=routing.worker_role,
                reason=f"revision using {routing.worker_role} role",
            )
            if self.settings.llm_backend == "ollama"
            else "echo"
        )
        budget.model_calls += 1
        _, terminated = self._check_budget(state, node="revise:model_call")
        if terminated:
            return terminated

        payload = {
            "user_request": state["message"],
            "system_instruction": state.get("system_prompt"),
            "execution_context": self.router.execution_context(),
            "task": self._current_task(state),
            "worker_result": state["worker_result"],
            "verification": state["verification"],
            "evidence": state.get("evidence", []),
        }
        result = (
            await WorkerAgent(self.settings, revision_model).revise(payload)
            if self.settings.llm_backend == "ollama"
            else WorkerResult.model_validate(state["worker_result"])
        )
        updated: AgentGraphState = {
            **state,
            "worker_result": result.model_dump(),
            "iterations": state.get("iterations", 0) + 1,
            "selected_models": self._with_selected_model(
                state, node="revise", model=revision_model
            ),
            "model": revision_model,
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

        if not self.settings.mcp_enabled:
            updated: AgentGraphState = {
                **state,
                "research_rounds": rounds,
                "termination_reason": "external evidence is required but MCP is disabled",
            }
            self._log_node(
                logging.WARNING,
                "graph_research_unavailable",
                node="research",
                state=updated,
                reason="mcp_disabled",
            )
            return updated
        if rounds > self.settings.phase2_max_research_rounds:
            return self._budget_termination(
                state,
                node="research:round_limit",
                exc=BudgetExceeded("maximum research rounds exceeded"),
            )

        metadata = dict(state.get("metadata", {}))
        verification = state.get("verification") or {}
        required_actions = verification.get("required_actions") or []
        candidates = self.router.select_tools(
            user_request=state["message"],
            task=self._current_task(state),
            required_actions=required_actions,
            inventory=self._inventory(state),
            metadata=metadata,
            limit=3,
        )
        log_kv(
            logger,
            logging.INFO,
            "graph_tool_candidates",
            request_id=state.get("request_id"),
            run_id=metadata.get("run_id"),
            task_id=self._current_task(state).get("id"),
            candidates=",".join(item.name for item in candidates),
            candidate_count=len(candidates),
            available_tool_count=len(self._inventory(state).tool_names),
        )
        if not candidates:
            updated: AgentGraphState = {
                **state,
                "research_rounds": rounds,
                "termination_reason": (
                    "external evidence is required, but no compatible read-only MCP "
                    "research tool is available"
                ),
            }
            self._log_node(
                logging.WARNING,
                "graph_research_no_tool",
                node="research",
                state=updated,
            )
            return updated

        failures: list[dict[str, str]] = []
        successful_tool: str | None = None
        successful_data: Any = None
        for candidate in candidates:
            log_kv(
                logger,
                logging.INFO,
                "graph_tool_selected",
                request_id=state.get("request_id"),
                run_id=metadata.get("run_id"),
                task_id=self._current_task(state).get("id"),
                tool=candidate.name,
                score=candidate.score,
                reason=candidate.reason,
                argument_keys=",".join(sorted(candidate.arguments)),
            )
            try:
                result = await self.tool_executor.execute(
                    candidate.name,
                    candidate.arguments,
                    budget=self._budget(state),
                    metadata=metadata,
                )
            except ToolApprovalRequired as exc:
                failures.append({"tool": candidate.name, "error": str(exc)})
                log_kv(
                    logger,
                    logging.WARNING,
                    "graph_tool_rejected",
                    tool=candidate.name,
                    reason="approval_required",
                )
                continue
            except BudgetExceeded as exc:
                return self._budget_termination(
                    state, node="research:tool_call", exc=exc
                )

            if result.ok:
                successful_tool = candidate.name
                successful_data = result.data
                log_kv(
                    logger,
                    logging.INFO,
                    "graph_research_result",
                    tool=candidate.name,
                    round=rounds,
                    ok=True,
                )
                break

            failures.append({"tool": candidate.name, "error": str(result.error or "tool failed")})
            log_kv(
                logger,
                logging.WARNING,
                "graph_research_result",
                tool=candidate.name,
                round=rounds,
                ok=False,
                error=result.error,
            )

        metadata.setdefault("research_failures", []).extend(failures)
        if successful_tool is None:
            updated = {
                **state,
                "metadata": metadata,
                "research_rounds": rounds,
                "termination_reason": (
                    "all compatible read-only MCP research tools failed; "
                    "no current evidence was retrieved"
                ),
            }
            self._log_node(
                logging.WARNING,
                "graph_research_failed",
                node="research",
                state=updated,
                attempted_tools=len(candidates),
            )
            return updated

        task_id = str(self._current_task(state).get("id") or f"task-{state.get('task_index', 0)}")
        evidence = metadata.setdefault("evidence", [])
        evidence.append(
            {
                "id": f"research-{task_id}-{rounds}",
                "source": successful_tool,
                "content": self._serialize_tool_data(successful_data),
                "task_id": task_id,
                "retrieved_at": self.router.execution_context()["execution_time_utc"],
            }
        )
        researched = list(dict.fromkeys([*(state.get("researched_task_ids") or []), task_id]))
        selected_tools = dict(state.get("selected_tools") or {})
        selected_tools[task_id] = successful_tool
        updated = {
            **state,
            "metadata": metadata,
            "research_rounds": rounds,
            "researched_task_ids": researched,
            "selected_tool": successful_tool,
            "selected_tools": selected_tools,
            "next_action": "worker",
        }
        self._log_node(
            logging.INFO,
            "graph_node_complete",
            node="research",
            state=updated,
            tool=successful_tool,
            evidence_items=len(evidence),
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
            updated: AgentGraphState = {
                **state,
                "replans": replans,
                "termination_reason": "maximum replans exceeded",
            }
            self._log_node(
                logging.WARNING,
                "graph_replan_limit_reached",
                node="replan",
                state=updated,
            )
            return updated

        budget = self._budget(state)
        planner_model = (
            self._select_model(
                state, node="replan", role="planner", reason="replanning role"
            )
            if self.settings.llm_backend == "ollama"
            else "echo"
        )
        budget.model_calls += 1
        _, terminated = self._check_budget(state, node="replan:model_call")
        if terminated:
            return terminated

        if self.settings.llm_backend == "ollama":
            planner = PlannerAgent(self.settings, planner_model)
            plan = await planner.invoke_json(
                system=PLANNER_PROMPT,
                payload={
                    "request": state["message"],
                    "runtime_context": {
                        **self.router.execution_context(),
                        "available_mcp_tools": self._inventory(state).tool_names,
                        "prior_plan": state.get("plan"),
                        "failed_task": self._current_task(state),
                        "verification": state.get("verification"),
                    },
                },
                schema=ExecutionPlan,
            )
        else:
            plan = ExecutionPlan.model_validate(state["plan"])

        updated = {
            **state,
            "plan": plan.model_dump(),
            "task_index": 0,
            "task_results": [],
            "worker_result": {},
            "verification": {},
            "research_rounds": 0,
            "replans": replans,
            "researched_task_ids": [],
            "selected_models": self._with_selected_model(
                state, node="replan", model=planner_model
            ),
        }
        updated = self._set_next_action(updated)
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
                "model": (state.get("selected_models") or {}).get("worker"),
                "tool": state.get("selected_tool"),
            }
        )
        updated: AgentGraphState = {
            **state,
            "task_results": results,
            "task_index": state.get("task_index", 0) + 1,
            "verification": {},
            "worker_result": {},
            "evidence": [],
            "research_rounds": 0,
            "selected_tool": None,
            "next_action": "worker",
        }
        tasks = (updated.get("plan") or {}).get("tasks", [])
        if updated.get("task_index", 0) < len(tasks):
            updated = self._set_next_action(updated)
        self._log_node(
            logging.INFO,
            "graph_task_advanced",
            node="advance",
            state=updated,
            completed_tasks=len(results),
            next_action=updated.get("next_action"),
        )
        return updated

    async def _finalize(self, state: AgentGraphState) -> AgentGraphState:
        self._log_node(logging.INFO, "graph_node_start", node="finalize", state=state)
        results = state.get("task_results", [])
        if state.get("termination_reason"):
            return await self._terminate(state)

        final_model = str(state.get("model") or self.settings.model_general)
        if self.settings.llm_backend == "ollama" and len(results) > 1:
            budget = self._budget(state)
            final_model = self._select_model(
                state,
                node="finalize",
                role="synthesis",
                reason="multi-task synthesis",
            )
            budget.model_calls += 1
            _, terminated = self._check_budget(state, node="finalize:model_call")
            if terminated:
                return await self._terminate(terminated)
            response = await SynthesizerAgent(
                self.settings, final_model
            ).synthesize(
                {
                    "user_request": state["message"],
                    "system_instruction": state.get("system_prompt"),
                    "execution_context": self.router.execution_context(),
                    "verified_results": results,
                }
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
            "model": final_model,
            "selected_models": self._with_selected_model(
                state, node="finalize", model=final_model
            ),
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
        request_id = str(uuid4())
        budget = ExecutionBudget(
            self.settings.execution_max_duration_seconds,
            self.settings.execution_max_model_calls,
            self.settings.execution_max_tool_calls,
            self.settings.execution_max_verifier_rounds,
        )
        history = await self.memory.get(thread_id)
        runtime_inventory = await self.inventory_service.load()
        metrics.inc("chat.requests")
        config = {"configurable": {"thread_id": thread_id}}
        log_kv(
            logger,
            logging.INFO,
            "graph_request_start",
            thread_id=thread_id,
            request_id=request_id,
            run_id=request.metadata.get("run_id"),
            backend=self.settings.llm_backend,
            max_duration_seconds=budget.max_duration_seconds,
            max_model_calls=budget.max_model_calls,
            max_tool_calls=budget.max_tool_calls,
            max_verifier_rounds=budget.max_verifier_rounds,
            history_messages=len(history),
            transient_state_reset=True,
        )
        log_kv(
            logger,
            logging.INFO,
            "graph_runtime_inventory",
            thread_id=thread_id,
            request_id=request_id,
            model_count=len(runtime_inventory.model_names),
            tool_count=len(runtime_inventory.tool_names),
            cached=runtime_inventory.cached,
            inventory_errors=",".join(sorted(runtime_inventory.errors)),
        )

        # LangGraph merges a new invocation into the prior checkpoint for the same
        # thread. Explicitly reset every per-run channel so conversation history
        # is retained while an old plan/result/termination reason cannot leak into
        # the new request.
        initial_state: AgentGraphState = build_fresh_run_state(
            message=request.message,
            system_prompt=request.system_prompt or self.settings.default_system_prompt,
            metadata=request.metadata,
            history=history,
            execution_budget=budget,
            request_id=request_id,
            inventory=runtime_inventory,
            backend=self.settings.llm_backend,
        )


        try:
            with span("chat.total"):
                result = await self.graph.ainvoke(initial_state, config=config)
        except BudgetExceeded as exc:
            metrics.inc("graph.unhandled_budget_exhausted")
            log_kv(
                logger,
                logging.ERROR,
                "graph_unhandled_budget_exhausted",
                thread_id=thread_id,
                request_id=request_id,
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
                "model": None,
                "termination_reason": str(exc),
                "plan": None,
                "task_results": [],
                "iterations": 0,
                "selected_models": {},
                "selected_tool": None,
                "selected_tools": {},
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
            request_id=request_id,
            run_id=request.metadata.get("run_id"),
            termination_reason=result.get("termination_reason"),
            response_chars=len(result["response"]),
            selected_tool=result.get("selected_tool"),
            selected_tools=json.dumps(result.get("selected_tools") or {}, sort_keys=True),
            selected_models=json.dumps(result.get("selected_models") or {}, sort_keys=True),
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
                "routing": result.get("routing"),
                "selected_models": result.get("selected_models"),
                "selected_tool": result.get("selected_tool"),
                "selected_tools": result.get("selected_tools"),
                "inventory": {
                    "models": runtime_inventory.model_names,
                    "tools": runtime_inventory.tool_names,
                    "errors": runtime_inventory.errors,
                    "cached": runtime_inventory.cached,
                },
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
