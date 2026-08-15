"""Outer orchestration for queued input, Runner execution, and Session SAVE."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from inspect import isawaitable
from typing import Literal

from agent_harness.checkpoints import (
    CheckpointConflictError,
    CheckpointCorruptError,
    CheckpointError,
    CheckpointStore,
    ContextCheckpoint,
    IncorporatedInput,
    RunnerCheckpoint,
)
from agent_harness.consolidation import ContextConsolidator
from agent_harness.context import (
    ContextBuilder,
    ResolvedSessionContext,
    SessionContextResolver,
)
from agent_harness.injection import (
    MessageInjectionBatch,
    MessageInjectionPoint,
)
from agent_harness.memory.dream import Dream
from agent_harness.messages import (
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from agent_harness.runner import (
    AgentRunner,
    AgentRunResult,
    AgentRunSpec,
    StreamEndHandler,
    TextDeltaHandler,
)
from agent_harness.runtime_io import (
    OutputSegmentEnded,
    OutputTextDelta,
    RuntimeRequest,
    RuntimeStreamEvent,
    RuntimeStreamHandler,
)
from agent_harness.session import PendingInput, Session, SessionHistoryConflictError
from agent_harness.storage.base import SessionStore
from agent_harness.token_estimation import PromptTokenEstimator
from agent_harness.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

RuntimeStatus = Literal[
    "idle",
    "completed",
    "failed",
    "cancelled",
    "limit_reached",
    "injected",
    "paused",
]
SideEffectStatus = Literal["none", "completed", "uncertain"]
PhaseResult = Literal["ok", "no_pending", "failed"]

INTERRUPTED_TOOL_CONTENT = (
    "Error: execution was interrupted after this tool call was accepted. "
    "Its completion state is unknown, so the harness did not replay it automatically."
)


class ExecutionPhase(Enum):
    LOAD = "load"
    PREPARE = "prepare"
    RUN = "run"
    SAVE = "save"
    RESPOND = "respond"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    """Public result of attempting to process one queued input."""

    session_id: str
    input_id: str | None
    status: RuntimeStatus
    final_content: str | None = None
    stop_reason: str | None = None
    tools_used: tuple[str, ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    error: str | None = None
    has_pending_continuation: bool = False
    remaining_pending_count: int = 0
    revision_target_input_id: str | None = None
    discarded_message_count: int = 0
    side_effect_status: SideEffectStatus = "none"
    discarded_tool_call_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _PausedRunState:
    """In-memory working prefix retained while one Session is paused for revision."""

    target_input_id: str
    messages: tuple[AgentMessage, ...]
    base_leaf_id: str | None
    save_cursor: int
    preserved_input_ids: tuple[str, ...]
    model_turn_offset: int
    tools_used: tuple[str, ...]
    usage: Mapping[str, int]
    discarded_message_count: int = 0
    side_effect_status: SideEffectStatus = "none"
    discarded_tool_call_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class ExecutionContext:
    """Minimal mutable state shared by Runtime phases."""

    session_id: str
    phase: ExecutionPhase = ExecutionPhase.LOAD
    pending_input: PendingInput | None = None
    base_leaf_id: str | None = None
    save_cursor: int = 0
    run_spec: AgentRunSpec | None = None
    run_result: AgentRunResult | None = None
    runtime_result: RuntimeResult | None = None
    checkpoint: ContextCheckpoint | None = None
    on_stream: RuntimeStreamHandler | None = None
    incorporated_inputs: list[PendingInput] = field(default_factory=list)
    model_seen_input_ids: set[str] = field(default_factory=set)
    remaining_pending_count: int = 0
    paused_run: _PausedRunState | None = None


@dataclass(slots=True)
class _ActiveRun:
    task: asyncio.Task[RuntimeResult]
    context: ExecutionContext
    pause_target_input_id: str | None = None
    pause_result: RuntimeResult | None = None


class RuntimeStateError(RuntimeError):
    """Runtime state changed incompatibly between orchestration phases."""


_TRANSITIONS: dict[tuple[ExecutionPhase, PhaseResult], ExecutionPhase] = {
    (ExecutionPhase.LOAD, "ok"): ExecutionPhase.PREPARE,
    (ExecutionPhase.LOAD, "no_pending"): ExecutionPhase.RESPOND,
    (ExecutionPhase.LOAD, "failed"): ExecutionPhase.RESPOND,
    (ExecutionPhase.PREPARE, "ok"): ExecutionPhase.RUN,
    (ExecutionPhase.PREPARE, "failed"): ExecutionPhase.RESPOND,
    (ExecutionPhase.RUN, "ok"): ExecutionPhase.SAVE,
    (ExecutionPhase.SAVE, "ok"): ExecutionPhase.RESPOND,
    (ExecutionPhase.RESPOND, "ok"): ExecutionPhase.DONE,
}


class AgentRuntime:
    """Advance one queued input through LOAD, PREPARE, RUN, SAVE, and RESPOND."""

    def __init__(
        self,
        *,
        runner: AgentRunner,
        session_store: SessionStore,
        context_builder: ContextBuilder,
        tools: ToolRegistry,
        model: str,
        max_turns: int = 20,
        extra_system_sections: Sequence[str] = (),
        consolidator: ContextConsolidator | None = None,
        context_resolver: SessionContextResolver | None = None,
        checkpoint_store: CheckpointStore | None = None,
        max_injected_inputs_per_run: int = 5,
        max_input_tokens: int | None = None,
        input_token_estimator: PromptTokenEstimator | None = None,
        dream: Dream | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty text")
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        if (
            type(max_injected_inputs_per_run) is not int
            or max_injected_inputs_per_run < 0
        ):
            raise ValueError("max_injected_inputs_per_run must be non-negative")
        if max_input_tokens is not None and (
            type(max_input_tokens) is not int or max_input_tokens <= 0
        ):
            raise ValueError("max_input_tokens must be a positive integer")
        self._runner = runner
        self._session_store = session_store
        self._context_builder = context_builder
        self._tools = tools
        self._model = model
        self._max_turns = max_turns
        self._extra_system_sections = tuple(extra_system_sections)
        self._consolidator = consolidator
        self._context_resolver = context_resolver or SessionContextResolver()
        self._checkpoint_store = checkpoint_store
        self._max_injected_inputs_per_run = max_injected_inputs_per_run
        self._max_input_tokens = max_input_tokens
        self._input_token_estimator = input_token_estimator
        self._dream = dream
        self._execution_locks: dict[str, asyncio.Lock] = {}
        self._active_runs: dict[str, _ActiveRun] = {}
        self._paused_sessions: set[str] = set()
        self._paused_runs: dict[str, _PausedRunState] = {}
        self._consolidation_tasks: dict[str, asyncio.Task[None]] = {}
        self._dream_task: asyncio.Task[None] | None = None
        self._dream_wakeup_requested = False

    def enqueue_input(
        self,
        session_id: str,
        source_message_id: str,
        content: str,
    ) -> PendingInput:
        """Persist user intent before any model execution begins."""

        self._validate_session_id(session_id)
        session = self._session_store.get_or_create(session_id)
        queued = session.enqueue(
            PendingInput(source_message_id=source_message_id, content=content)
        )
        self._session_store.save(session)
        return queued

    async def submit(
        self,
        request: RuntimeRequest,
        *,
        on_stream: RuntimeStreamHandler | None = None,
    ) -> RuntimeResult:
        if not isinstance(request, RuntimeRequest):
            raise TypeError("request must be a RuntimeRequest")
        queued = self.enqueue_input(
            request.session_id,
            request.source_message_id,
            request.content,
        )
        return await self.run_next(
            request.session_id,
            on_stream=on_stream,
            requested_input_id=queued.id,
        )

    # Execution supervision

    async def run_next(
        self,
        session_id: str,
        *,
        on_stream: RuntimeStreamHandler | None = None,
        requested_input_id: str | None = None,
    ) -> RuntimeResult:
        """Run at most the current queue head under a per-Session execution lock."""

        self._validate_session_id(session_id)
        lock = self._execution_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            short_circuit = self._run_short_circuit_result(
                session_id,
                requested_input_id,
            )
            if short_circuit is not None:
                return short_circuit
            return await self._run_execution_context(session_id, on_stream)

    def _run_short_circuit_result(
        self,
        session_id: str,
        requested_input_id: str | None,
    ) -> RuntimeResult | None:
        if session_id in self._paused_sessions:
            session = self._session_store.get_or_create(session_id)
            paused = self._paused_runs.get(session_id)
            return RuntimeResult(
                session_id=session_id,
                input_id=requested_input_id,
                status="paused",
                stop_reason="session_paused_for_revision",
                remaining_pending_count=len(session.pending_inputs),
                revision_target_input_id=(
                    paused.target_input_id if paused is not None else None
                ),
            )
        if requested_input_id is None:
            return None
        session = self._session_store.get_or_create(session_id)
        if requested_input_id in {item.id for item in session.pending_inputs}:
            return None
        return RuntimeResult(
            session_id=session_id,
            input_id=requested_input_id,
            status="injected",
            stop_reason="injected_into_active_run",
        )

    async def _run_execution_context(
        self,
        session_id: str,
        on_stream: RuntimeStreamHandler | None,
    ) -> RuntimeResult:
        context = ExecutionContext(session_id=session_id, on_stream=on_stream)
        active = self._register_active_run(context)
        try:
            await self._drive_execution(context, active)
            if active.pause_target_input_id is not None:
                active.pause_result = self._capture_paused_run(active)
                return active.pause_result
            if context.runtime_result is None:
                raise RuntimeStateError("DONE reached without a RuntimeResult")
            return context.runtime_result
        finally:
            if self._active_runs.get(session_id) is active:
                del self._active_runs[session_id]

    def _register_active_run(self, context: ExecutionContext) -> _ActiveRun:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeStateError("run_next requires an asyncio Task")
        active = _ActiveRun(task=task, context=context)
        self._active_runs[context.session_id] = active
        return active

    async def _drive_execution(
        self,
        context: ExecutionContext,
        active: _ActiveRun,
    ) -> None:
        try:
            while context.phase is not ExecutionPhase.DONE:
                phase_result = await self._execute_phase(context)
                if active.pause_target_input_id is not None:
                    return
                context.phase = self._next_phase(context.phase, phase_result)
        except asyncio.CancelledError:
            if active.pause_target_input_id is None:
                raise

    @staticmethod
    def _next_phase(phase: ExecutionPhase, result: PhaseResult) -> ExecutionPhase:
        try:
            return _TRANSITIONS[(phase, result)]
        except KeyError as exc:
            raise RuntimeStateError(
                f"Invalid Runtime transition: {phase.value}/{result}"
            ) from exc

    # Interactive pause and revision

    async def pause_for_revision(
        self,
        session_id: str,
        input_id: str | None = None,
    ) -> RuntimeResult:
        """Stop one active run and retain only the prefix before the revised input."""

        self._validate_session_id(session_id)
        session = self._session_store.get_or_create(session_id)
        pending = tuple(session.pending_inputs)
        if not pending:
            return self._completed_before_pause_result(session_id)
        if input_id is not None and input_id not in {item.id for item in pending}:
            raise RuntimeStateError(f"PendingInput is not editable: {input_id}")
        repeated_pause = self._repeated_pause_result(session, input_id)
        if repeated_pause is not None:
            return repeated_pause

        target_id = input_id or pending[-1].id
        self._paused_sessions.add(session_id)
        active = self._active_runs.get(session_id)
        if active is None:
            return self._pause_inactive_session(session, target_id)
        return await self._pause_active_run(session_id, target_id, active)

    def _repeated_pause_result(
        self,
        session: Session,
        input_id: str | None,
    ) -> RuntimeResult | None:
        existing = self._paused_runs.get(session.id)
        if existing is None:
            return None
        if input_id is not None and input_id != existing.target_input_id:
            raise RuntimeStateError("Session is already paused for another PendingInput")
        return self._paused_result(session, existing, stop_reason="already_paused")

    def _pause_inactive_session(
        self,
        session: Session,
        target_id: str,
    ) -> RuntimeResult:
        state = _PausedRunState(
            target_input_id=target_id,
            messages=tuple(session.copy_history()),
            base_leaf_id=session.active_leaf_id,
            save_cursor=len(session.active_messages()),
            preserved_input_ids=(),
            model_turn_offset=0,
            tools_used=(),
            usage={},
        )
        self._delete_checkpoint_for_revision(session.id)
        self._paused_runs[session.id] = state
        return self._paused_result(session, state)

    async def _pause_active_run(
        self,
        session_id: str,
        target_id: str,
        active: _ActiveRun,
    ) -> RuntimeResult:
        active.pause_target_input_id = target_id
        active.task.cancel()
        outcomes = await asyncio.gather(active.task, return_exceptions=True)
        if active.pause_result is not None:
            return active.pause_result
        if outcomes and isinstance(outcomes[0], Exception):
            raise outcomes[0]

        refreshed = self._session_store.get_or_create(session_id)
        if target_id not in {item.id for item in refreshed.pending_inputs}:
            self._paused_sessions.discard(session_id)
            return self._completed_before_pause_result(session_id, target_id)
        raise RuntimeStateError("Active run ended without materializing paused state")

    @staticmethod
    def _completed_before_pause_result(
        session_id: str,
        input_id: str | None = None,
    ) -> RuntimeResult:
        return RuntimeResult(
            session_id=session_id,
            input_id=input_id,
            status="idle",
            stop_reason="completed_before_pause",
        )

    def revise_paused_input(
        self,
        session_id: str,
        input_id: str,
        content: str,
    ) -> PendingInput:
        """Persist a new revision for the input selected by pause_for_revision()."""

        self._validate_session_id(session_id)
        paused = self._paused_runs.get(session_id)
        if session_id not in self._paused_sessions or paused is None:
            raise RuntimeStateError("Session is not paused for revision")
        if input_id != paused.target_input_id:
            raise RuntimeStateError("Only the selected revision target may be edited")
        session = self._session_store.get_or_create(session_id)
        with session.rollback_on_error():
            edited = session.edit_pending(input_id, content)
            self._session_store.save(session)
        return edited

    async def restart_pending(
        self,
        session_id: str,
        *,
        on_stream: RuntimeStreamHandler | None = None,
    ) -> RuntimeResult:
        """Leave paused mode and continue from the retained working prefix."""

        self._validate_session_id(session_id)
        if session_id not in self._paused_sessions:
            raise RuntimeStateError("Session is not paused for revision")
        self._paused_sessions.remove(session_id)
        return await self.run_next(session_id, on_stream=on_stream)

    async def revise_and_restart(
        self,
        session_id: str,
        input_id: str,
        content: str,
        *,
        on_stream: RuntimeStreamHandler | None = None,
    ) -> RuntimeResult:
        self.revise_paused_input(session_id, input_id, content)
        return await self.restart_pending(session_id, on_stream=on_stream)

    def discard_paused_run(self, session_id: str) -> bool:
        """Drop in-memory pause control, for example before an explicit Session clear."""

        self._validate_session_id(session_id)
        if session_id in self._active_runs:
            raise RuntimeStateError("Cannot discard pause control while a run is active")
        was_paused = session_id in self._paused_sessions
        self._paused_sessions.discard(session_id)
        self._paused_runs.pop(session_id, None)
        return was_paused

    async def _execute_phase(self, context: ExecutionContext) -> PhaseResult:
        if context.phase is ExecutionPhase.LOAD:
            return self._load(context)
        if context.phase is ExecutionPhase.PREPARE:
            return await self._prepare(context)
        if context.phase is ExecutionPhase.RUN:
            return await self._run(context)
        if context.phase is ExecutionPhase.SAVE:
            return self._save(context)
        if context.phase is ExecutionPhase.RESPOND:
            return self._respond(context)
        raise RuntimeStateError(f"Cannot execute Runtime phase: {context.phase.value}")

    # Runtime phases

    def _load(self, context: ExecutionContext) -> PhaseResult:
        session = self._session_store.get_or_create(context.session_id)
        if not session.pending_inputs:
            self._delete_checkpoint_best_effort(context.session_id)
            return "no_pending"
        context.pending_input = session.pending_inputs[0]
        paused = self._paused_runs.get(context.session_id)
        if paused is not None:
            context.paused_run = paused
            context.base_leaf_id = paused.base_leaf_id
            context.save_cursor = paused.save_cursor
            self._delete_checkpoint_best_effort(context.session_id)
            return "ok"
        context.base_leaf_id = session.active_leaf_id
        context.save_cursor = len(session.active_messages())
        store = self._checkpoint_store
        if store is None:
            return "ok"
        try:
            checkpoint = store.load(context.session_id)
        except CheckpointCorruptError as exc:
            self._set_checkpoint_failure(context, "checkpoint_corrupt", exc)
            return "failed"
        except CheckpointError as exc:
            self._set_checkpoint_failure(context, "checkpoint_error", exc)
            return "failed"
        if checkpoint is None:
            return "ok"

        pending = context.pending_input
        if checkpoint.input_id != pending.id:
            pending_ids = {item.id for item in session.pending_inputs}
            if checkpoint.input_id not in pending_ids:
                self._delete_checkpoint_best_effort(context.session_id)
                return "ok"
            self._set_checkpoint_failure(
                context,
                "checkpoint_conflict",
                CheckpointConflictError("Checkpoint does not belong to the pending queue head"),
            )
            return "failed"
        if checkpoint.input_revision != pending.revision:
            self._delete_checkpoint_best_effort(context.session_id)
            return "ok"
        context.checkpoint = checkpoint
        return "ok"

    async def _prepare(self, context: ExecutionContext) -> PhaseResult:
        pending = self._require_pending(context)
        session = self._session_store.get_or_create(context.session_id)
        self._validate_prepare_session(context, session, pending)
        if self._prepare_incorporated_inputs(context, session) == "failed":
            return "failed"
        if await self._enforce_prepare_input_limit(context, session) == "failed":
            return "failed"

        await self._maybe_consolidate(context, session, pending)
        session = self._session_store.get_or_create(context.session_id)
        self._validate_prepare_session(context, session, pending)
        resolved = self._context_resolver.resolve(session)
        working_messages = self._build_working_messages(context, session)
        context.run_spec = self._build_run_spec(
            context,
            pending,
            resolved,
            working_messages,
        )
        if context.paused_run is not None:
            self._paused_runs.pop(context.session_id, None)
        return "ok"

    # PREPARE helpers

    @staticmethod
    def _validate_prepare_session(
        context: ExecutionContext,
        session: Session,
        pending: PendingInput,
    ) -> None:
        if session.active_leaf_id != context.base_leaf_id:
            raise SessionHistoryConflictError(
                "Session active leaf changed between LOAD and PREPARE"
            )
        if len(session.active_messages()) != context.save_cursor:
            raise SessionHistoryConflictError(
                "Session active branch changed between LOAD and PREPARE"
            )
        if not session.pending_inputs or session.pending_inputs[0] != pending:
            raise RuntimeStateError("Pending queue head changed between LOAD and PREPARE")

    def _prepare_incorporated_inputs(
        self,
        context: ExecutionContext,
        session: Session,
    ) -> PhaseResult:
        checkpoint = context.checkpoint
        paused = context.paused_run
        if checkpoint is not None:
            conflict = self._checkpoint_conflict(checkpoint, session)
            if conflict is not None:
                self._set_checkpoint_failure(context, "checkpoint_conflict", conflict)
                return "failed"

        if paused is not None:
            self._restore_paused_inputs(context, session, paused)
            return "ok"
        if checkpoint is None:
            context.incorporated_inputs = [self._require_pending(context)]
            return "ok"
        return self._restore_checkpoint_inputs(context, session, checkpoint)

    async def _enforce_prepare_input_limit(
        self,
        context: ExecutionContext,
        session: Session,
    ) -> PhaseResult:
        if self._max_input_tokens is None:
            return "ok"
        for item in context.incorporated_inputs:
            if item.id in context.model_seen_input_ids:
                continue
            estimated = await self._estimate_input_tokens(item)
            if estimated <= self._max_input_tokens:
                continue
            context.runtime_result = RuntimeResult(
                session_id=context.session_id,
                input_id=self._require_pending(context).id,
                status="failed",
                stop_reason="input_too_large",
                error=(
                    f"PendingInput {item.id} is estimated at {estimated} tokens; "
                    f"the per-input limit is {self._max_input_tokens}"
                ),
                remaining_pending_count=len(session.pending_inputs),
                revision_target_input_id=item.id,
            )
            return "failed"
        return "ok"

    async def _estimate_input_tokens(self, item: PendingInput) -> int:
        estimator = self._input_token_estimator
        if estimator is not None:
            try:
                result = estimator.estimate(
                    model=self._model,
                    system_prompt=None,
                    messages=ContextBuilder.build_messages((item.to_user_message(),)),
                    tools=(),
                )
                value = await result if isawaitable(result) else result
                if type(value) is int and value > 0:
                    return value
            except Exception:
                logger.debug(
                    "Input token estimation failed for %s; using UTF-8 byte upper bound",
                    item.id,
                    exc_info=True,
                )
        return max(1, len(item.content.encode("utf-8")))

    def _restore_paused_inputs(
        self,
        context: ExecutionContext,
        session: Session,
        paused: _PausedRunState,
    ) -> None:
        context.incorporated_inputs = list(
            self._pending_prefix_by_ids(session, paused.preserved_input_ids)
        )
        preserved_users = {
            message.id: message
            for message in paused.messages
            if isinstance(message, UserMessage)
        }
        if any(
            preserved_users.get(item.id) != item.to_user_message()
            for item in context.incorporated_inputs
        ):
            raise RuntimeStateError(
                "A preserved PendingInput changed outside the selected revision target"
            )
        context.model_seen_input_ids.update(paused.preserved_input_ids)
        if not context.incorporated_inputs:
            self._claim_pending_suffix(context, session, limit=1)
            return
        available = self._remaining_injection_capacity(context)
        if available > 0:
            self._claim_pending_suffix(context, session, limit=available)

    def _restore_checkpoint_inputs(
        self,
        context: ExecutionContext,
        session: Session,
        checkpoint: ContextCheckpoint,
    ) -> PhaseResult:
        context.incorporated_inputs = list(
            self._pending_prefix_for_checkpoint(session, checkpoint)
        )
        context.model_seen_input_ids.update(
            item.id for item in context.incorporated_inputs
        )
        if checkpoint.next_model_turn <= self._max_turns:
            return "ok"
        self._set_checkpoint_failure(
            context,
            "checkpoint_conflict",
            CheckpointConflictError(
                "Checkpoint model turn exceeds the configured max_turns"
            ),
        )
        return "failed"

    async def _maybe_consolidate(
        self,
        context: ExecutionContext,
        session: Session,
        pending: PendingInput,
    ) -> None:
        if (
            self._consolidator is None
            or context.checkpoint is not None
            or context.paused_run is not None
        ):
            return
        background = self._consolidation_tasks.get(context.session_id)
        if background is not None:
            await asyncio.shield(background)
        await self._consolidator.maybe_consolidate(
            session,
            pending_message=pending.to_user_message(),
            context_builder=self._context_builder,
            tools=self._tools,
            extra_system_sections=self._extra_system_sections,
        )
        self._schedule_dream()

    def _schedule_background_consolidation(self, session_id: str) -> None:
        if self._consolidator is None:
            return
        existing = self._consolidation_tasks.get(session_id)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(self._run_background_consolidation(session_id))
        self._consolidation_tasks[session_id] = task
        task.add_done_callback(
            lambda completed, key=session_id: self._forget_consolidation_task(
                key, completed
            )
        )

    async def _run_background_consolidation(self, session_id: str) -> None:
        consolidator = self._consolidator
        if consolidator is None:
            return
        try:
            session = self._session_store.get_or_create(session_id)
            await consolidator.maybe_consolidate(
                session,
                context_builder=self._context_builder,
                tools=self._tools,
                extra_system_sections=self._extra_system_sections,
            )
            self._schedule_dream()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Background context consolidation failed for %s", session_id)

    def _forget_consolidation_task(
        self,
        session_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if self._consolidation_tasks.get(session_id) is task:
            self._consolidation_tasks.pop(session_id, None)

    async def wait_for_background_tasks(self) -> None:
        """Wait until all scheduled consolidation and Dream work has settled."""

        while True:
            consolidation_tasks = tuple(self._consolidation_tasks.items())
            dream_task = self._dream_task
            tasks = [task for _session_id, task in consolidation_tasks]
            if dream_task is not None:
                tasks.append(dream_task)
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)
            for session_id, task in consolidation_tasks:
                self._forget_consolidation_task(session_id, task)
            if (
                self._dream_task is dream_task
                and dream_task is not None
                and dream_task.done()
            ):
                self._dream_task = None

    def _schedule_dream(self) -> None:
        if self._dream is None:
            return
        task = self._dream_task
        if task is not None and not task.done():
            self._dream_wakeup_requested = True
            return
        self._dream_wakeup_requested = False
        task = asyncio.create_task(self._run_dream_until_idle())
        self._dream_task = task
        task.add_done_callback(self._forget_dream_task)

    async def _run_dream_until_idle(self) -> None:
        dream = self._dream
        if dream is None:
            return
        try:
            while True:
                self._dream_wakeup_requested = False
                result = await dream.run()
                if not result.did_work:
                    if self._dream_wakeup_requested:
                        continue
                    return
                if result.stop_reason != "completed":
                    return
                # A completed max-size batch may leave more work. Continue
                # until the durable Inbox reports empty.
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Background Dream processing failed")

    def _forget_dream_task(self, task: asyncio.Task[None]) -> None:
        if self._dream_task is task:
            self._dream_task = None

    async def cancel_background_consolidation(self, session_id: str) -> bool:
        """Cancel one Session's proactive consolidation before destructive mutation."""

        self._validate_session_id(session_id)
        task = self._consolidation_tasks.get(session_id)
        if task is None:
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    def _build_working_messages(
        self,
        context: ExecutionContext,
        session: Session,
    ) -> list[AgentMessage]:
        paused = context.paused_run
        if paused is not None:
            messages = list(paused.messages)
            messages.extend(
                item.to_user_message()
                for item in context.incorporated_inputs[
                    len(paused.preserved_input_ids) :
                ]
            )
            return messages

        messages = session.copy_history()
        checkpoint = context.checkpoint
        if checkpoint is None:
            messages.extend(
                item.to_user_message() for item in context.incorporated_inputs
            )
            return messages

        messages.extend(checkpoint.messages)
        self._append_interrupted_tool_results(messages, checkpoint)
        self._append_resumed_inputs(messages, context, session, checkpoint)
        return messages

    @staticmethod
    def _append_interrupted_tool_results(
        messages: list[AgentMessage],
        checkpoint: ContextCheckpoint,
    ) -> None:
        if checkpoint.phase != "awaiting_tools":
            return
        assistant = messages[-1]
        if not isinstance(assistant, AssistantMessage):
            raise RuntimeStateError(
                "Validated awaiting_tools Checkpoint has no AssistantMessage"
            )
        messages.extend(
            ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                content=INTERRUPTED_TOOL_CONTENT,
                is_error=True,
            )
            for call in assistant.tool_calls
        )

    def _append_resumed_inputs(
        self,
        messages: list[AgentMessage],
        context: ExecutionContext,
        session: Session,
        checkpoint: ContextCheckpoint,
    ) -> None:
        if checkpoint.phase == "final_response":
            return
        available = self._remaining_injection_capacity(context)
        if available <= 0:
            return
        resumed = self._claim_pending_suffix(context, session, limit=available)
        messages.extend(item.to_user_message() for item in resumed)

    def _remaining_injection_capacity(self, context: ExecutionContext) -> int:
        return self._max_injected_inputs_per_run - (
            len(context.incorporated_inputs) - 1
        )

    def _build_run_spec(
        self,
        context: ExecutionContext,
        pending: PendingInput,
        resolved: ResolvedSessionContext,
        working_messages: Sequence[AgentMessage],
    ) -> AgentRunSpec:
        checkpoint = context.checkpoint
        paused = context.paused_run
        on_text_delta, on_stream_end = self._build_stream_callbacks(context, pending)
        model_turn_offset = 0
        initial_tools_used: Sequence[str] = ()
        initial_usage: Mapping[str, int] = {}
        if checkpoint is not None:
            model_turn_offset = checkpoint.next_model_turn
            initial_tools_used = checkpoint.tools_used
            initial_usage = checkpoint.usage
        elif paused is not None:
            model_turn_offset = paused.model_turn_offset
            initial_tools_used = paused.tools_used
            initial_usage = paused.usage
        return AgentRunSpec(
            initial_messages=working_messages,
            tools=self._tools,
            model=checkpoint.model if checkpoint is not None else self._model,
            system_prompt=self._context_builder.build_system_prompt(
                context_summary=(
                    resolved.summary.content if resolved.summary is not None else None
                ),
                extra_sections=self._extra_system_sections,
            ),
            max_turns=self._max_turns,
            stream=context.on_stream is not None,
            on_text_delta=on_text_delta,
            on_stream_end=on_stream_end,
            model_message_start=resolved.covered_message_count,
            context_prefix_messages=self._context_builder.build_skill_context_prefix(
                working_messages[: resolved.covered_message_count]
            ),
            current_turn_start=context.save_cursor,
            checkpoint_callback=(
                partial(self._persist_runner_checkpoint, context)
                if self._checkpoint_store is not None
                else None
            ),
            model_turn_offset=model_turn_offset,
            initial_tools_used=initial_tools_used,
            initial_usage=initial_usage,
            injection_callback=partial(self._collect_pending_injection, context),
            max_injected_inputs_per_run=self._max_injected_inputs_per_run,
            initial_injected_input_count=len(context.incorporated_inputs) - 1,
        )

    def _build_stream_callbacks(
        self,
        context: ExecutionContext,
        pending: PendingInput,
    ) -> tuple[TextDeltaHandler | None, StreamEndHandler | None]:
        handler = context.on_stream
        if handler is None:
            return None, None
        segment_index = 0

        async def emit(event: RuntimeStreamEvent) -> None:
            callback_result = handler(event)
            if isawaitable(callback_result):
                await callback_result

        async def on_text_delta(text: str) -> None:
            await emit(
                OutputTextDelta(
                    input_id=pending.id,
                    segment_index=segment_index,
                    text=text,
                )
            )

        async def on_stream_end(resuming: bool) -> None:
            nonlocal segment_index
            await emit(
                OutputSegmentEnded(
                    input_id=pending.id,
                    segment_index=segment_index,
                    resuming=resuming,
                )
            )
            segment_index += 1

        return on_text_delta, on_stream_end

    def _persist_runner_checkpoint(
        self,
        context: ExecutionContext,
        runner_checkpoint: RunnerCheckpoint,
    ) -> None:
        store = self._checkpoint_store
        if store is None:
            return
        current_session = self._session_store.get_or_create(context.session_id)
        current_pending = self._require_pending(context)
        conflict = self._checkpoint_identity_conflict(
            current_session,
            context.incorporated_inputs,
            base_leaf_id=context.base_leaf_id,
            save_cursor=context.save_cursor,
        )
        if conflict is not None:
            raise conflict
        durable = ContextCheckpoint(
            session_id=context.session_id,
            input_id=current_pending.id,
            input_revision=current_pending.revision,
            base_leaf_id=context.base_leaf_id,
            save_cursor=context.save_cursor,
            phase=runner_checkpoint.phase,
            model=runner_checkpoint.model,
            next_model_turn=runner_checkpoint.next_model_turn,
            messages=tuple(runner_checkpoint.messages[context.save_cursor :]),
            incorporated_inputs=tuple(
                IncorporatedInput(id=item.id, revision=item.revision)
                for item in context.incorporated_inputs
            ),
            tools_used=runner_checkpoint.tools_used,
            usage=runner_checkpoint.usage,
            terminal_status=runner_checkpoint.terminal_status,
            stop_reason=runner_checkpoint.stop_reason,
            final_content=runner_checkpoint.final_content,
        )
        store.save(durable)
        context.checkpoint = durable

    async def _collect_pending_injection(
        self,
        context: ExecutionContext,
        point: MessageInjectionPoint,
        limit: int,
    ) -> MessageInjectionBatch:
        current_session = self._session_store.get_or_create(context.session_id)
        conflict = self._checkpoint_identity_conflict(
            current_session,
            context.incorporated_inputs,
            base_leaf_id=context.base_leaf_id,
            save_cursor=context.save_cursor,
        )
        if conflict is not None:
            raise conflict
        accepted = limit
        if self._max_input_tokens is not None:
            start = len(context.incorporated_inputs)
            candidates = current_session.pending_inputs[start : start + limit]
            accepted = 0
            for item in candidates:
                if await self._estimate_input_tokens(item) > self._max_input_tokens:
                    break
                accepted += 1
        claimed = self._claim_pending_suffix(context, current_session, limit=accepted)
        return MessageInjectionBatch(
            point=point,
            messages=tuple(item.to_user_message() for item in claimed),
        )

    async def _run(self, context: ExecutionContext) -> PhaseResult:
        if context.run_spec is None:
            raise RuntimeStateError("RUN requires an AgentRunSpec")
        checkpoint = context.checkpoint
        if checkpoint is not None and checkpoint.phase == "final_response":
            if (
                checkpoint.terminal_status is None
                or checkpoint.stop_reason is None
                or checkpoint.final_content is None
            ):
                raise RuntimeStateError("final_response Checkpoint has no terminal state")
            context.run_result = AgentRunResult(
                final_content=checkpoint.final_content,
                messages=tuple(context.run_spec.initial_messages),
                status=checkpoint.terminal_status,
                stop_reason=checkpoint.stop_reason,
                tools_used=checkpoint.tools_used,
                usage=dict(checkpoint.usage),
                model_seen_user_message_ids=tuple(
                    item.id for item in checkpoint.incorporated_inputs
                ),
            )
            context.model_seen_input_ids.update(
                item.id for item in checkpoint.incorporated_inputs
            )
            return "ok"
        context.run_result = await self._runner.run(context.run_spec)
        context.model_seen_input_ids.update(
            context.run_result.model_seen_user_message_ids
        )
        return "ok"

    def _save(self, context: ExecutionContext) -> PhaseResult:
        result = self._require_run_result(context)
        if result.status not in {"completed", "limit_reached"}:
            return "ok"

        incorporated_ids = tuple(item.id for item in context.incorporated_inputs)
        unseen_ids = set(incorporated_ids).difference(context.model_seen_input_ids)
        if unseen_ids:
            raise RuntimeStateError(
                "SAVE cannot consume PendingInput that was not included in a Provider request: "
                + ", ".join(sorted(unseen_ids))
            )

        session = self._session_store.get_or_create(context.session_id)
        with session.rollback_on_error():
            session.commit_working_messages(
                working_messages=result.messages,
                save_cursor=context.save_cursor,
                base_leaf_id=context.base_leaf_id,
                consumed_input_ids=incorporated_ids,
            )
            self._session_store.save(session)
        context.remaining_pending_count = len(session.pending_inputs)
        self._delete_checkpoint_best_effort(context.session_id)
        self._schedule_background_consolidation(context.session_id)
        return "ok"

    def _respond(self, context: ExecutionContext) -> PhaseResult:
        if context.runtime_result is not None:
            return "ok"
        if context.pending_input is None:
            context.runtime_result = RuntimeResult(
                session_id=context.session_id,
                input_id=None,
                status="idle",
                stop_reason="no_pending_input",
            )
            return "ok"

        result = self._require_run_result(context)
        context.runtime_result = RuntimeResult(
            session_id=context.session_id,
            input_id=context.pending_input.id,
            status=result.status,
            final_content=result.final_content,
            stop_reason=result.stop_reason,
            tools_used=result.tools_used,
            usage=dict(result.usage),
            error=result.error,
            has_pending_continuation=context.remaining_pending_count > 0,
            remaining_pending_count=context.remaining_pending_count,
        )
        return "ok"

    # Session and Checkpoint invariants

    @staticmethod
    def _checkpoint_identity_conflict(
        session: Session,
        incorporated_inputs: Sequence[PendingInput],
        *,
        base_leaf_id: str | None,
        save_cursor: int,
    ) -> CheckpointConflictError | None:
        if session.active_leaf_id != base_leaf_id:
            return CheckpointConflictError("Session Active Leaf changed during execution")
        if len(session.active_messages()) != save_cursor:
            return CheckpointConflictError("Session history changed during execution")
        incorporated = tuple(incorporated_inputs)
        if not incorporated:
            return CheckpointConflictError("Execution has no incorporated PendingInput")
        pending_prefix = tuple(session.pending_inputs[: len(incorporated)])
        if pending_prefix != incorporated:
            return CheckpointConflictError("PendingInput prefix changed during execution")
        return None

    def _checkpoint_conflict(
        self,
        checkpoint: ContextCheckpoint,
        session: Session,
    ) -> CheckpointConflictError | None:
        try:
            incorporated = self._pending_prefix_for_checkpoint(session, checkpoint)
        except CheckpointConflictError as exc:
            return exc
        conflict = self._checkpoint_identity_conflict(
            session,
            incorporated,
            base_leaf_id=checkpoint.base_leaf_id,
            save_cursor=checkpoint.save_cursor,
        )
        if conflict is not None:
            return conflict
        pending_by_id = {item.id: item for item in session.pending_inputs}
        checkpoint_users = {
            message.id: message
            for message in checkpoint.messages
            if isinstance(message, UserMessage)
        }
        if any(
            checkpoint_users.get(item.id) != pending_by_id[item.id].to_user_message()
            for item in checkpoint.incorporated_inputs
        ):
            return CheckpointConflictError(
                "Checkpoint contains stale PendingInput content"
            )
        return None

    @staticmethod
    def _pending_prefix_for_checkpoint(
        session: Session,
        checkpoint: ContextCheckpoint,
    ) -> tuple[PendingInput, ...]:
        count = len(checkpoint.incorporated_inputs)
        prefix = tuple(session.pending_inputs[:count])
        expected = tuple(
            (item.id, item.revision) for item in checkpoint.incorporated_inputs
        )
        actual = tuple((item.id, item.revision) for item in prefix)
        if actual != expected:
            raise CheckpointConflictError(
                "Checkpoint incorporated inputs do not match the pending queue prefix"
            )
        return prefix

    @staticmethod
    def _claim_pending_suffix(
        context: ExecutionContext,
        session: Session,
        *,
        limit: int,
    ) -> tuple[PendingInput, ...]:
        if type(limit) is not int or limit < 0:
            raise ValueError("message injection limit must be non-negative")
        known = tuple(context.incorporated_inputs)
        if tuple(session.pending_inputs[: len(known)]) != known:
            raise CheckpointConflictError(
                "PendingInput prefix changed before message injection"
            )
        claimed = tuple(session.pending_inputs[len(known) : len(known) + limit])
        context.incorporated_inputs.extend(claimed)
        return claimed

    @staticmethod
    def _pending_prefix_by_ids(
        session: Session,
        input_ids: Sequence[str],
    ) -> tuple[PendingInput, ...]:
        expected = tuple(input_ids)
        prefix = tuple(session.pending_inputs[: len(expected)])
        if tuple(item.id for item in prefix) != expected:
            raise RuntimeStateError(
                "Paused working prefix no longer matches the PendingInput queue"
            )
        return prefix

    def _capture_paused_run(self, active: _ActiveRun) -> RuntimeResult:
        context = active.context
        target_id = active.pause_target_input_id
        if target_id is None:
            raise RuntimeStateError("Paused run has no revision target")
        session = self._session_store.get_or_create(context.session_id)
        if target_id not in {item.id for item in session.pending_inputs}:
            self._paused_sessions.discard(context.session_id)
            return self._completed_before_pause_result(context.session_id, target_id)

        messages, usage = self._pause_source(context, session)
        preserved, discarded = self._split_messages_at_input(messages, target_id)
        preserved_input_ids, tools_used, model_turn_offset = self._preserved_metadata(
            context,
            session,
            preserved,
        )
        side_effect_status, discarded_calls = self._discarded_tool_effects(discarded)

        state = _PausedRunState(
            target_input_id=target_id,
            messages=preserved,
            base_leaf_id=context.base_leaf_id,
            save_cursor=context.save_cursor,
            preserved_input_ids=preserved_input_ids,
            model_turn_offset=model_turn_offset,
            tools_used=tools_used,
            usage=usage,
            discarded_message_count=len(discarded),
            side_effect_status=side_effect_status,
            discarded_tool_call_ids=discarded_calls,
        )
        self._delete_checkpoint_for_revision(context.session_id)
        self._paused_runs[context.session_id] = state
        return self._paused_result(session, state)

    # Pause materialization helpers

    @staticmethod
    def _pause_source(
        context: ExecutionContext,
        session: Session,
    ) -> tuple[tuple[AgentMessage, ...], dict[str, int]]:
        if context.run_result is not None:
            return context.run_result.messages, dict(context.run_result.usage)
        if context.run_spec is not None:
            return tuple(context.run_spec.initial_messages), dict(
                context.run_spec.initial_usage
            )
        return tuple(session.copy_history()), {}

    @staticmethod
    def _split_messages_at_input(
        messages: Sequence[AgentMessage],
        target_id: str,
    ) -> tuple[tuple[AgentMessage, ...], tuple[AgentMessage, ...]]:
        target_index = next(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, UserMessage) and message.id == target_id
            ),
            None,
        )
        copied = tuple(messages)
        if target_index is None:
            return copied, ()
        return copied[:target_index], copied[target_index:]

    @staticmethod
    def _preserved_metadata(
        context: ExecutionContext,
        session: Session,
        preserved: Sequence[AgentMessage],
    ) -> tuple[tuple[str, ...], tuple[str, ...], int]:
        current_turn = tuple(preserved[context.save_cursor :])
        pending_ids = {item.id for item in session.pending_inputs}
        input_ids = tuple(
            message.id
            for message in current_turn
            if isinstance(message, UserMessage) and message.id in pending_ids
        )
        tools_used = tuple(
            call.name
            for message in current_turn
            if isinstance(message, AssistantMessage)
            for call in message.tool_calls
        )
        model_turn_offset = sum(
            isinstance(message, AssistantMessage) for message in current_turn
        )
        return input_ids, tools_used, model_turn_offset

    @staticmethod
    def _discarded_tool_effects(
        discarded: Sequence[AgentMessage],
    ) -> tuple[SideEffectStatus, tuple[str, ...]]:
        call_ids = tuple(
            call.id
            for message in discarded
            if isinstance(message, AssistantMessage)
            for call in message.tool_calls
        )
        if not call_ids:
            return "none", ()
        completed_ids = {
            message.tool_call_id
            for message in discarded
            if isinstance(message, ToolResultMessage)
        }
        if any(call_id not in completed_ids for call_id in call_ids):
            return "uncertain", call_ids
        return "completed", call_ids

    @staticmethod
    def _paused_result(
        session: Session,
        state: _PausedRunState,
        *,
        stop_reason: str = "user_revision",
    ) -> RuntimeResult:
        return RuntimeResult(
            session_id=session.id,
            input_id=state.target_input_id,
            status="paused",
            stop_reason=stop_reason,
            tools_used=state.tools_used,
            usage=dict(state.usage),
            remaining_pending_count=len(session.pending_inputs),
            revision_target_input_id=state.target_input_id,
            discarded_message_count=state.discarded_message_count,
            side_effect_status=state.side_effect_status,
            discarded_tool_call_ids=state.discarded_tool_call_ids,
        )

    def _delete_checkpoint_best_effort(self, session_id: str) -> None:
        if self._checkpoint_store is None:
            return
        try:
            self._checkpoint_store.delete(session_id)
        except CheckpointError:
            pass

    def _delete_checkpoint_for_revision(self, session_id: str) -> None:
        if self._checkpoint_store is not None:
            self._checkpoint_store.delete(session_id)

    @staticmethod
    def _set_checkpoint_failure(
        context: ExecutionContext,
        stop_reason: str,
        error: Exception,
    ) -> None:
        context.runtime_result = RuntimeResult(
            session_id=context.session_id,
            input_id=(context.pending_input.id if context.pending_input is not None else None),
            status="failed",
            stop_reason=stop_reason,
            error=f"{type(error).__name__}: {error}",
        )

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty text")

    @staticmethod
    def _require_pending(context: ExecutionContext) -> PendingInput:
        if context.pending_input is None:
            raise RuntimeStateError(f"{context.phase.value} requires a PendingInput")
        return context.pending_input

    @staticmethod
    def _require_run_result(context: ExecutionContext) -> AgentRunResult:
        if context.run_result is None:
            raise RuntimeStateError(f"{context.phase.value} requires an AgentRunResult")
        return context.run_result


__all__ = [
    "AgentRuntime",
    "ExecutionContext",
    "ExecutionPhase",
    "RuntimeResult",
    "RuntimeRequest",
    "RuntimeStateError",
    "RuntimeStatus",
    "SideEffectStatus",
    "INTERRUPTED_TOOL_CONTENT",
]
