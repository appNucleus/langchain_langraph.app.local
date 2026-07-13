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
    ConversationGate,
    ResumeRunNotFoundError,
    RunConflictError,
    RunIdentityService,
    RunRegistry,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.execution import BudgetExceeded, ExecutionBudget
from app.schemas.run import RunIdentity
from app.services.routing import build_fresh_run_state
from app.settings import Settings

logger = logging.getLogger(__name__)


class ChatRuntimeAgent(ChatAgent):
    """Chat runtime with distinct conversation, run, and checkpoint identities."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.identity_service = RunIdentityService(settings)
        self.conversation_gate = ConversationGate()
        self.run_registry = RunRegistry(
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
            checkpoint_exists = await self._checkpoint_exists_safely(
                config,
                required=identity.resumed or identity.client_supplied_run_id,
            )
            if identity.resumed and not checkpoint_exists:
                raise ResumeRunNotFoundError(
                    "No checkpoint exists for the requested resumable run"
                )
            if (
                not identity.resumed
                and identity.client_supplied_run_id
                and checkpoint_exists
            ):
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
        self._log_request_start(
            identity=identity,
            request_id=request_id,
            budget=budget,
            history_count=len(history),
            resumed=identity.resumed,
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
                    "resume_requested": identity.resume_requested,
                    "resumed": identity.resumed,
                }
            )

        result = await self._invoke_graph_safely(
            graph_input=graph_input,
            config=config,
            identity=identity,
            request_id=request_id,
            budget=budget,
        )
        effective_budget = result.get("execution_budget")
        if not isinstance(effective_budget, ExecutionBudget):
            effective_budget = budget

        history_persisted = await self._append_history_safely(
            identity=identity,
            existing_history=history,
            user_message=request.message,
            assistant_message=str(result.get("response") or ""),
        )

        resume_token = self.identity_service.issue_resume_token(identity)
        response = ChatResponse.from_result(
            thread_id=identity.conversation_id,
            conversation_id=identity.conversation_id,
            run_id=identity.run_id,
            response=str(result.get("response") or self._fallback_message("empty response")),
            backend=str(result.get("backend") or self.settings.llm_backend),
            model=result.get("model"),
            metadata={
                **self.identity_service.sanitized_metadata(request),
                "runtime_contract": "run-identity-v1",
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
                    "history_persisted": history_persisted,
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
            history_persisted=history_persisted,
            selected_tool=result.get("selected_tool"),
            selected_tools=json.dumps(result.get("selected_tools") or {}, sort_keys=True),
            selected_models=json.dumps(result.get("selected_models") or {}, sort_keys=True),
            **self._budget_fields(effective_budget),
        )
        return response

    async def _invoke_graph_safely(
        self,
        *,
        graph_input: AgentGraphState | None,
        config: dict[str, Any],
        identity: RunIdentity,
        request_id: str,
        budget: ExecutionBudget,
    ) -> dict[str, Any]:
        try:
            with span("chat.total"):
                result = await self.graph.ainvoke(graph_input, config=config)
            if not isinstance(result, dict):
                raise RuntimeError("The graph returned a non-object result")
            return result
        except BudgetExceeded as exc:
            reason = str(exc)
            metrics.inc("graph.unhandled_budget_exhausted")
            self._log_runtime_failure(
                "graph_budget_exhausted",
                exc,
                identity=identity,
                request_id=request_id,
                budget=budget,
            )
            return self._safe_failure_result(reason=reason)
        except StructuredOutputError as exc:
            reason = (
                f"structured output failed for {exc.schema_name} after "
                f"{len(exc.attempted_models)} bounded attempt(s)"
            )
            metrics.inc("graph.unhandled_structured_output_error")
            self._log_runtime_failure(
                "graph_structured_output_error",
                exc,
                identity=identity,
                request_id=request_id,
                budget=budget,
            )
            return self._safe_failure_result(reason=reason)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = f"agent execution failed: {type(exc).__name__}"
            metrics.inc("graph.unhandled_runtime_error")
            self._log_runtime_failure(
                "graph_runtime_error",
                exc,
                identity=identity,
                request_id=request_id,
                budget=budget,
            )
            return self._safe_failure_result(reason=reason)

    async def _checkpoint_exists_safely(
        self,
        config: dict[str, Any],
        *,
        required: bool,
    ) -> bool:
        try:
            checkpointer = self.state_runtime.checkpointer
            async_get = getattr(checkpointer, "aget_tuple", None)
            if callable(async_get):
                return await async_get(config) is not None
            get = getattr(checkpointer, "get_tuple", None)
            if callable(get):
                return await asyncio.to_thread(get, config) is not None
            raise RuntimeError("The configured checkpointer cannot inspect run identity")
        except Exception:
            logger.exception("checkpoint_identity_lookup_failed required=%s", required)
            if required:
                raise
            metrics.inc("chat.checkpoint_lookup_degraded")
            return False

    async def _append_history_safely(
        self,
        *,
        identity: RunIdentity,
        existing_history: list[dict[str, Any]],
        user_message: str,
        assistant_message: str,
    ) -> bool:
        try:
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
                return True
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
            return True
        except Exception:
            metrics.inc("chat.history_append_error")
            logger.exception(
                "conversation_history_append_failed conversation_id=%s run_id=%s",
                identity.conversation_id,
                identity.run_id,
            )
            return False

    def _safe_failure_result(self, *, reason: str) -> dict[str, Any]:
        return {
            "response": self._fallback_message(reason),
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

    @staticmethod
    def _fallback_message(reason: str) -> str:
        return (
            "Execution stopped safely before a verified answer could be produced: "
            f"{reason}."
        )

    def _log_request_start(
        self,
        *,
        identity: RunIdentity,
        request_id: str,
        budget: ExecutionBudget,
        history_count: int,
        resumed: bool,
    ) -> None:
        log_kv(
            logger,
            logging.INFO,
            "graph_request_start",
            thread_id=identity.conversation_id,
            conversation_id=identity.conversation_id,
            run_id=identity.run_id,
            execution_thread_id=identity.execution_thread_id,
            request_id=request_id,
            resumed=resumed,
            backend=self.settings.llm_backend,
            history_messages=history_count,
            **self._budget_fields(budget),
        )

    def _log_runtime_failure(
        self,
        event_name: str,
        exc: BaseException,
        *,
        identity: RunIdentity,
        request_id: str,
        budget: ExecutionBudget,
    ) -> None:
        logger.error(
            "%s conversation_id=%s run_id=%s execution_thread_id=%s request_id=%s "
            "budget=%s error=%r",
            event_name,
            identity.conversation_id,
            identity.run_id,
            identity.execution_thread_id,
            request_id,
            self._budget_fields(budget),
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )

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
            resume_token_expires_in_seconds=self.identity_service.token_ttl_seconds,
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
