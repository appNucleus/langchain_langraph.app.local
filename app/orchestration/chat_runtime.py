from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from app.agents.base import StructuredOutputError
from app.graph import ChatAgent
from app.graphs.state import AgentGraphState
from app.logging_config import log_kv
from app.observability.events import event
from app.observability.metrics import metrics
from app.observability.tracing import span
from app.orchestration.run_identity import (
    ConversationBusyError,
    ConversationGate,
    ResumeRunNotFoundError,
    ResumeTokenRevokedError,
    RunConflictError,
    RunNotResumableError,
    RunIdentityService,
    StaleLeaseError,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.execution import BudgetExceeded, ExecutionBudget
from app.schemas.run import RunIdentity
from app.services.routing import build_fresh_run_state
from app.settings import Settings
from app.state.run_repository import RunLease, RunRecord, RunRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RunExecution:
    response: ChatResponse
    status: Literal["completed", "failed"]
    termination_reason: str | None
    error_code: str | None


class ChatRuntimeAgent(ChatAgent):
    """Chat runtime with durable run identity and restart-safe outcomes."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.identity_service = RunIdentityService(settings)
        self.conversation_gate = ConversationGate()
        self.run_repository: RunRepository = self.state_runtime.runs

    async def start(self) -> None:
        await super().start()
        self.run_repository = self.state_runtime.runs

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
        # The local gate is only a latency optimization. RunRepository.acquire is
        # the correctness boundary across workers and application processes.
        async with self.conversation_gate.hold(identity):
            config = identity.langgraph_config()
            checkpoint_exists: bool | None = None
            resume_migrated = False
            if identity.resumed:
                record = await self.run_repository.get(identity.run_id)
                if record is None:
                    # Additive Stage 2 rollout: a valid token plus an existing
                    # checkpoint may migrate a pre-run-repository execution.
                    checkpoint_exists = await self._checkpoint_exists_safely(
                        config,
                        required=True,
                    )
                    if not checkpoint_exists:
                        raise ResumeRunNotFoundError(
                            "No checkpoint exists for the requested resumable run"
                        )
                    record, created = await self.run_repository.create_or_get(identity)
                    resume_migrated = True
                else:
                    if (
                        record.conversation_id != identity.conversation_id
                        or record.execution_thread_id != identity.execution_thread_id
                        or record.request_hash != identity.request_hash
                        or record.request_hash_version != identity.request_hash_version
                        or record.state_schema_version != identity.state_schema_version
                    ):
                        raise RunConflictError(
                            "The resumable run identity does not match the durable record"
                        )
                    created = False
            else:
                record, created = await self.run_repository.create_or_get(identity)
            replay = await self._preflight_existing_run(
                request=request,
                identity=identity,
                record=record,
                created=created,
                resume_migrated=resume_migrated,
            )
            if replay is not None:
                return replay

            if checkpoint_exists is None:
                checkpoint_exists = await self._checkpoint_exists_safely(
                    config,
                    required=(
                        identity.resumed
                        or (identity.client_supplied_run_id and not created)
                    ),
                )
            if identity.resumed and not checkpoint_exists:
                raise ResumeRunNotFoundError(
                    "No checkpoint exists for the requested resumable run"
                )
            if (
                not identity.resumed
                and identity.client_supplied_run_id
                and created
                and checkpoint_exists
            ):
                raise RunConflictError(
                    "The supplied run_id already has checkpoint state; use resume=true "
                    "with its server-issued resume token"
                )

            owner_id = str(uuid4())
            lease = await self.run_repository.acquire(
                identity,
                owner_id=owner_id,
                ttl_seconds=self.settings.run_lease_ttl_seconds,
            )
            terminal_persisted = False
            try:
                execution = await self._execute_with_heartbeat(
                    request=request,
                    identity=identity,
                    config=config,
                    lease=lease,
                )
                checkpoint_id = await self._checkpoint_id_safely(config)
                await self.run_repository.mark_terminal(
                    lease,
                    status=execution.status,
                    response_payload=execution.response.model_dump(mode="json"),
                    termination_reason=execution.termination_reason,
                    error_code=execution.error_code,
                    checkpoint_id=checkpoint_id,
                )
                terminal_persisted = True

                history_persisted = await self._append_history_safely(
                    identity=identity,
                    user_message=request.message,
                    assistant_message=execution.response.response,
                )
                execution.response.metadata.setdefault("persistence", {})[
                    "history_persisted"
                ] = history_persisted
                if history_persisted:
                    await self.run_repository.mark_history_committed(
                        lease,
                        response_payload=execution.response.model_dump(mode="json"),
                    )
                return execution.response
            except asyncio.CancelledError:
                await self._mark_interrupted_safely(
                    lease,
                    termination_reason="request_cancelled",
                    error_code="request_cancelled",
                )
                metrics.inc("chat.run_interrupted")
                raise
            except StaleLeaseError:
                metrics.inc("chat.stale_lease")
                raise
            except Exception:
                if not terminal_persisted:
                    await self._mark_interrupted_safely(
                        lease,
                        termination_reason="runtime_interrupted",
                        error_code="runtime_interrupted",
                    )
                metrics.inc("chat.run_failed")
                raise
            finally:
                await self.run_repository.release(lease)

    async def _preflight_existing_run(
        self,
        *,
        request: ChatRequest,
        identity: RunIdentity,
        record: RunRecord,
        created: bool,
        resume_migrated: bool,
    ) -> ChatResponse | None:
        if identity.resumed and identity.resume_token_version != record.resume_token_version:
            raise ResumeTokenRevokedError("The resume token has been revoked")

        if record.response_payload is not None and record.status in {
            "completed",
            "failed",
            "cancelled",
        }:
            response = ChatResponse.model_validate(record.response_payload)
            if record.history_committed_at is None:
                committed = await self._append_history_safely(
                    identity=identity,
                    user_message=request.message,
                    assistant_message=response.response,
                )
                response.metadata.setdefault("persistence", {})[
                    "history_persisted"
                ] = committed
                if committed:
                    await self.run_repository.reconcile_history_committed(
                        identity.run_id,
                        response_payload=response.model_dump(mode="json"),
                    )
            metrics.inc("chat.idempotent_durable_replay")
            return response

        if identity.resumed and resume_migrated and record.status == "pending":
            return None

        if created:
            if identity.resumed:
                raise ResumeRunNotFoundError(
                    "No durable run record exists for the requested resumable run"
                )
            return None

        lease_expired = (
            record.lease_expires_at is not None
            and record.lease_expires_at <= datetime.now(UTC)
        )
        if identity.resumed:
            if record.status == "interrupted":
                return None
            if record.status == "running" and lease_expired:
                # A crashed owner cannot mark itself interrupted. The expired
                # durable lease is the fail-closed proof that takeover is safe.
                return None
            raise RunNotResumableError(
                f"Run status {record.status!r} cannot be resumed"
            )

        if record.status == "running":
            if lease_expired:
                raise RunConflictError(
                    "The previous owner lease expired; resume with the server-issued token"
                )
            raise ConversationBusyError("The run is already active")
        if record.status == "pending":
            # A pending record may be safely retried when no lease was acquired.
            return None
        raise RunConflictError(
            "The run already exists; use its resume token only when it is interrupted"
        )

    async def _execute_with_heartbeat(
        self,
        *,
        request: ChatRequest,
        identity: RunIdentity,
        config: dict[str, Any],
        lease: RunLease,
    ) -> _RunExecution:
        execution_task = asyncio.create_task(
            self._execute_run(request, identity, config=config)
        )
        heartbeat_task = asyncio.create_task(self._lease_heartbeat(lease))
        try:
            done, _ = await asyncio.wait(
                {execution_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                error = heartbeat_task.exception()
                execution_task.cancel()
                await asyncio.gather(execution_task, return_exceptions=True)
                if error is not None:
                    raise error
                raise StaleLeaseError("The conversation lease heartbeat stopped")
            return await execution_task
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
            if not execution_task.done():
                execution_task.cancel()
                await asyncio.gather(execution_task, return_exceptions=True)

    async def _lease_heartbeat(self, lease: RunLease) -> None:
        current = lease
        while True:
            await asyncio.sleep(self.settings.run_lease_heartbeat_seconds)
            current = await self.run_repository.renew(
                current,
                ttl_seconds=self.settings.run_lease_ttl_seconds,
            )
            metrics.inc("chat.lease_renewed")

    async def _execute_run(
        self,
        request: ChatRequest,
        identity: RunIdentity,
        *,
        config: dict[str, Any],
    ) -> _RunExecution:
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
                "request_hash_version": identity.request_hash_version,
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

        resume_token = self.identity_service.issue_resume_token(identity)
        status: Literal["completed", "failed"] = (
            "failed" if result.get("_run_status") == "failed" else "completed"
        )
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
                "run_repository_contract": "durable-run-v1",
                "identity": {
                    "conversation_id": identity.conversation_id,
                    "run_id": identity.run_id,
                    "execution_thread_id": identity.execution_thread_id,
                    "checkpoint_namespace": identity.checkpoint_namespace,
                    "state_schema_version": identity.state_schema_version,
                    "request_hash_version": identity.request_hash_version,
                    "resumed": identity.resumed,
                    "resume_token": resume_token,
                    "resume_token_key_id": self.identity_service.active_key_id,
                    "resume_token_version": identity.resume_token_version,
                    "resume_token_expires_in_seconds": (
                        self.identity_service.token_ttl_seconds
                    ),
                    "resume_token_persistent_across_restart": (
                        self.identity_service.token_persistent
                    ),
                },
                "persistence": {
                    "state_backend": self.settings.state_backend,
                    "run_repository_backend": self.settings.run_repository_backend,
                    "checkpoint_backend": self.settings.checkpoint_backend,
                    "artifact_backend": self.settings.artifact_backend,
                    "history_persisted": False,
                },
                "run_status": status,
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
            run_status=status,
            termination_reason=result.get("termination_reason"),
            response_chars=len(response.response),
            selected_tool=result.get("selected_tool"),
            selected_tools=json.dumps(result.get("selected_tools") or {}, sort_keys=True),
            selected_models=json.dumps(result.get("selected_models") or {}, sort_keys=True),
            **self._budget_fields(effective_budget),
        )
        return _RunExecution(
            response=response,
            status=status,
            termination_reason=result.get("termination_reason"),
            error_code=result.get("_run_error_code"),
        )

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
            return self._safe_failure_result(
                reason=reason,
                error_code="budget_exhausted",
            )
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
            return self._safe_failure_result(
                reason=reason,
                error_code="structured_output_failed",
            )
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
            return self._safe_failure_result(
                reason=reason,
                error_code="agent_execution_failed",
            )

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

    async def _checkpoint_id_safely(self, config: dict[str, Any]) -> str | None:
        try:
            checkpointer = self.state_runtime.checkpointer
            async_get = getattr(checkpointer, "aget_tuple", None)
            checkpoint = (
                await async_get(config)
                if callable(async_get)
                else await asyncio.to_thread(checkpointer.get_tuple, config)
            )
            checkpoint_config = getattr(checkpoint, "config", None) if checkpoint else None
            if not isinstance(checkpoint_config, dict):
                return None
            configurable = checkpoint_config.get("configurable") or {}
            value = configurable.get("checkpoint_id")
            return str(value) if value else None
        except Exception:
            metrics.inc("chat.checkpoint_id_lookup_degraded")
            return None

    async def _append_history_safely(
        self,
        *,
        identity: RunIdentity,
        user_message: str,
        assistant_message: str,
    ) -> bool:
        try:
            await self.memory.append_turn(
                identity.conversation_id,
                run_id=identity.run_id,
                user_message={
                    "role": "user",
                    "content": user_message,
                    "metadata": {
                        "conversation_id": identity.conversation_id,
                        "run_id": identity.run_id,
                    },
                },
                assistant_message={
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

    async def _mark_interrupted_safely(
        self,
        lease: RunLease,
        *,
        termination_reason: str,
        error_code: str,
    ) -> None:
        try:
            await self.run_repository.mark_terminal(
                lease,
                status="interrupted",
                response_payload=None,
                termination_reason=termination_reason,
                error_code=error_code,
            )
        except StaleLeaseError:
            metrics.inc("chat.interrupt_stale_lease")
        except Exception:
            logger.exception("run_interruption_persist_failed run_id=%s", lease.run_id)

    def _safe_failure_result(
        self,
        *,
        reason: str,
        error_code: str,
    ) -> dict[str, Any]:
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
            "_run_status": "failed",
            "_run_error_code": error_code,
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
