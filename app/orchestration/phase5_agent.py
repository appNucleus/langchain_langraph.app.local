from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from app.agents.base import StructuredOutputError
from app.graph import ChatAgent
from app.graphs.state import AgentGraphState
from app.logging_config import log_kv
from app.observability.events import event
from app.observability.metrics import metrics
from app.observability.tracing import span
from app.orchestration.run_identity import (
    LocalConversationGate,
    LocalRunRegistry,
    ResumeRunNotFoundError,
    RunConflictError,
    RunIdentityService,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.execution import BudgetExceeded, ExecutionBudget
from app.schemas.run import RunIdentity
from app.services.routing import build_fresh_run_state
from app.settings import Settings

logger = logging.getLogger(__name__)


class Phase5ChatAgent(ChatAgent):
    """Chat runtime with separated conversation and graph-run identities."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.identity_service = RunIdentityService(settings)
        self.conversation_gate = LocalConversationGate()
        self.run_registry = LocalRunRegistry(
            ttl_seconds=min(
                settings.state_ttl_seconds,
                settings.resume_token_ttl_seconds,
            ),
            max_records=settings.state_max_sessions,
        )

    @staticmethod
    def _task_fields(state: AgentGraphState) -> dict[str, object]:
        fields = ChatAgent._task_fields(state)
        metadata = state.get("metadata") or {}
        fields.update(
            {
                "request_id": state.get("request_id"),
                "conversation_id": state.get("conversation_id")
                or metadata.get("conversation_id"),
                "run_id": state.get("run_id") or metadata.get("run_id"),
                "execution_thread_id": state.get("execution_thread_id")
                or metadata.get("execution_thread_id"),
            }
        )
        return fields

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        identity = self.identity_service.normalize(request)
        return await self._ainvoke_with_identity(request, identity)

    async def _ainvoke_with_identity(
        self,
        request: ChatRequest,
        identity: RunIdentity,
    ) -> ChatResponse:
        async with self.conversation_gate.hold(identity):
            cached = await self.run_registry.cached_response(identity)
            if cached is not None:
                metrics.inc("chat.idempotent_cache_hit")
                return cached

            config = identity.langgraph_config()
            checkpoint_exists = await self._checkpoint_exists(config)
            if identity.resumed:
                if not checkpoint_exists:
                    raise ResumeRunNotFoundError(
                        "No checkpoint exists for the requested resumable run"
                    )
            elif identity.client_supplied_run_id and checkpoint_exists:
                raise RunConflictError(
                    "The supplied run_id already has checkpoint state; use resume=true "
                    "with its server-issued resume token"
                )

            await self.run_registry.start(identity)
            try:
                response = await self._execute_run(request, identity, config=config)
            except asyncio.CancelledError:
                await self.run_registry.interrupt(identity)
                metrics.inc("chat.run_interrupted")
                raise
            except Exception:
                await self.run_registry.interrupt(identity)
                metrics.inc("chat.run_failed")
                raise
            await self.run_registry.complete(identity, response)
            return response

    async def _execute_run(
        self,
        request: ChatRequest,
        identity: RunIdentity,
        *,
        config: dict[str, Any],
    ) -> ChatResponse:
        request_id = str(uuid4())
        budget = ExecutionBudget(
            self.settings.execution_max_duration_seconds,
            self.settings.execution_max_model_calls,
            self.settings.execution_max_tool_calls,
            self.settings.execution_max_verifier_rounds,
        )
        history = await self.memory.get(identity.conversation_id)
        runtime_inventory = await self.inventory_service.load()
        system_decision = self.router.prepare_system_prompt(
            message=request.message,
            provided=request.system_prompt,
        )
        metadata = self.identity_service.sanitized_metadata(request)
        metadata.update(
            {
                "conversation_id": identity.conversation_id,
                "run_id": identity.run_id,
                "execution_thread_id": identity.execution_thread_id,
                "state_schema_version": identity.state_schema_version,
                "resume_requested": identity.resume_requested,
            }
        )

        metrics.inc("chat.requests")
        log_kv(
            logger,
            logging.INFO,
            "graph_request_start",
            thread_id=identity.conversation_id,
            conversation_id=identity.conversation_id,
            run_id=identity.run_id,
            execution_thread_id=identity.execution_thread_id,
            request_id=request_id,
            resumed=identity.resumed,
            backend=self.settings.llm_backend,
            max_duration_seconds=budget.max_duration_seconds,
            max_model_calls=budget.max_model_calls,
            max_tool_calls=budget.max_tool_calls,
            max_verifier_rounds=budget.max_verifier_rounds,
            history_messages=len(history),
            transient_state_reset=not identity.resumed,
        )
        log_kv(
            logger,
            logging.INFO,
            "graph_runtime_inventory",
            thread_id=identity.conversation_id,
            conversation_id=identity.conversation_id,
            run_id=identity.run_id,
            execution_thread_id=identity.execution_thread_id,
            request_id=request_id,
            model_count=len(runtime_inventory.model_names),
            tool_count=len(runtime_inventory.tool_names),
            cached=runtime_inventory.cached,
            inventory_errors=",".join(sorted(runtime_inventory.errors)),
        )
        log_kv(
            logger,
            logging.INFO,
            "graph_system_prompt_prepared",
            thread_id=identity.conversation_id,
            conversation_id=identity.conversation_id,
            run_id=identity.run_id,
            execution_thread_id=identity.execution_thread_id,
            request_id=request_id,
            source=system_decision.source,
            domain=system_decision.domain,
            prompt_chars=len(system_decision.prompt),
            requires_external_evidence=system_decision.requires_external_evidence,
        )

        graph_input: AgentGraphState | None
        if identity.resumed:
            graph_input = None
        else:
            graph_input = build_fresh_run_state(
                message=request.message,
                system_prompt=system_decision.prompt,
                system_prompt_source=system_decision.source,
                request_domain=system_decision.domain,
                metadata=metadata,
                history=history,
                execution_budget=budget,
                request_id=request_id,
                inventory=runtime_inventory,
                backend=self.settings.llm_backend,
            )
            graph_input.update(
                {
                    "conversation_id": identity.conversation_id,
                    "run_id": identity.run_id,
                    "execution_thread_id": identity.execution_thread_id,
                    "state_schema_version": identity.state_schema_version,
                    "resume_requested": False,
                    "resumed": False,
                }
            )

        try:
            with span("chat.total"):
                result = await self.graph.ainvoke(graph_input, config=config)
            if not isinstance(result, dict):
                raise ResumeRunNotFoundError(
                    "The graph did not return resumable state for this run"
                )
        except BudgetExceeded as exc:
            metrics.inc("graph.unhandled_budget_exhausted")
            log_kv(
                logger,
                logging.ERROR,
                "graph_unhandled_budget_exhausted",
                thread_id=identity.conversation_id,
                conversation_id=identity.conversation_id,
                run_id=identity.run_id,
                execution_thread_id=identity.execution_thread_id,
                request_id=request_id,
                reason=str(exc),
                **self._budget_fields(budget),
            )
            result = self._safe_failure_result(
                reason=str(exc),
                response=(
                    "Execution stopped safely before a verified answer could be "
                    f"produced: {exc}."
                ),
            )
        except StructuredOutputError as exc:
            metrics.inc("graph.unhandled_structured_output_error")
            reason = (
                f"structured output failed for {exc.schema_name} after "
                f"{len(exc.attempted_models)} bounded attempt(s)"
            )
            log_kv(
                logger,
                logging.ERROR,
                "graph_unhandled_structured_output_error",
                thread_id=identity.conversation_id,
                conversation_id=identity.conversation_id,
                run_id=identity.run_id,
                execution_thread_id=identity.execution_thread_id,
                request_id=request_id,
                schema=exc.schema_name,
                primary_model=exc.primary_model,
                attempted_models=",".join(exc.attempted_models),
                reason=reason,
            )
            result = self._safe_failure_result(
                reason=reason,
                response=(
                    "Execution stopped safely before a verified answer could be "
                    f"produced: {reason}."
                ),
            )

        effective_budget = result.get("execution_budget")
        if not isinstance(effective_budget, ExecutionBudget):
            effective_budget = budget

        await self._append_history_once(
            identity=identity,
            existing_history=history,
            user_message=request.message,
            assistant_message=str(result["response"]),
        )

        resume_token = self.identity_service.issue_resume_token(identity)
        response = ChatResponse.from_result(
            thread_id=identity.conversation_id,
            conversation_id=identity.conversation_id,
            run_id=identity.run_id,
            response=str(result["response"]),
            backend=str(result["backend"]),
            model=result.get("model"),
            metadata={
                **self.identity_service.sanitized_metadata(request),
                "phase": "5",
                "identity": {
                    "conversation_id": identity.conversation_id,
                    "run_id": identity.run_id,
                    "execution_thread_id": identity.execution_thread_id,
                    "checkpoint_namespace": identity.checkpoint_namespace,
                    "state_schema_version": identity.state_schema_version,
                    "resumed": identity.resumed,
                    "resume_token": resume_token,
                    "resume_token_expires_in_seconds": (
                        self.identity_service.token_ttl_seconds
                    ),
                    "resume_token_persistent_across_restart": (
                        self.identity_service.token_persistent
                    ),
                },
                "persistence": {
                    "state_backend": self.settings.state_backend,
                    "checkpoint_backend": self.settings.checkpoint_backend,
                    "artifact_backend": self.settings.artifact_backend,
                },
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
                "usage": {
                    "model_calls": effective_budget.model_calls,
                    "tool_calls": effective_budget.tool_calls,
                    "verifier_rounds": effective_budget.verifier_rounds,
                    "elapsed_seconds": round(effective_budget.elapsed_seconds, 3),
                },
            },
        )
        log_kv(
            logger,
            logging.INFO,
            "graph_request_complete",
            thread_id=identity.conversation_id,
            conversation_id=identity.conversation_id,
            run_id=identity.run_id,
            execution_thread_id=identity.execution_thread_id,
            request_id=request_id,
            resumed=identity.resumed,
            termination_reason=result.get("termination_reason"),
            response_chars=len(response.response),
            selected_tool=result.get("selected_tool"),
            selected_tools=json.dumps(result.get("selected_tools") or {}, sort_keys=True),
            selected_models=json.dumps(result.get("selected_models") or {}, sort_keys=True),
            **self._budget_fields(effective_budget),
        )
        return response

    async def _checkpoint_exists(
        self,
        config: dict[str, Any],
    ) -> bool:
        checkpointer = self.state_runtime.checkpointer
        async_get = getattr(checkpointer, "aget_tuple", None)
        if callable(async_get):
            return await async_get(config) is not None
        get = getattr(checkpointer, "get_tuple", None)
        if callable(get):
            return await asyncio.to_thread(get, config) is not None
        raise RuntimeError("The configured checkpointer cannot inspect run identity")

    async def _append_history_once(
        self,
        *,
        identity: RunIdentity,
        existing_history: list[dict[str, Any]],
        user_message: str,
        assistant_message: str,
    ) -> None:
        if any(
            str(
                item.get("run_id")
                or (
                    item.get("metadata", {}).get("run_id")
                    if isinstance(item.get("metadata"), dict)
                    else ""
                )
                or ""
            )
            == identity.run_id
            for item in existing_history
            if isinstance(item, dict)
        ):
            metrics.inc("chat.history_idempotent_skip")
            return
        await self.memory.append(
            identity.conversation_id,
            {
                "role": "user",
                "content": user_message,
                "metadata": {
                    "conversation_id": identity.conversation_id,
                    "run_id": identity.run_id,
                },
            },
            {
                "role": "assistant",
                "content": assistant_message,
                "metadata": {
                    "conversation_id": identity.conversation_id,
                    "run_id": identity.run_id,
                },
            },
        )

    def _safe_failure_result(self, *, reason: str, response: str) -> dict[str, Any]:
        return {
            "response": response,
            "backend": self.settings.llm_backend,
            "model": None,
            "termination_reason": reason,
            "plan": None,
            "task_results": [],
            "iterations": 0,
            "selected_models": {},
            "selected_tool": None,
            "selected_tools": {},
            "research_queries": {},
        }

    async def astream_events(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[dict[str, object]]:
        identity = self.identity_service.normalize(request)
        resume_token = self.identity_service.issue_resume_token(identity)
        yield event(
            "request_started",
            thread_id=identity.conversation_id,
            conversation_id=identity.conversation_id,
            run_id=identity.run_id,
            execution_thread_id=identity.execution_thread_id,
            resumed=identity.resumed,
            resume_token=resume_token,
            resume_token_expires_in_seconds=(
                self.identity_service.token_ttl_seconds
            ),
        )
        yield event(
            "planning_started",
            conversation_id=identity.conversation_id,
            run_id=identity.run_id,
        )
        task = asyncio.create_task(self._ainvoke_with_identity(request, identity))
        try:
            while not task.done():
                await asyncio.sleep(0.5)
                yield event(
                    "working",
                    stage="agent_graph",
                    conversation_id=identity.conversation_id,
                    run_id=identity.run_id,
                )
            response = await task
            yield event("completed", response=response.model_dump(mode="json"))
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
