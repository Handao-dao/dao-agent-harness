"""Interactive CLI adapter for DAO Agent Harness."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Callable, Coroutine, Sequence
from pathlib import Path
from uuid import uuid4

from agent_harness.artifacts import ArtifactPolicy, LocalArtifactStore
from agent_harness.checkpoints import JsonFileCheckpointStore
from agent_harness.consolidation import (
    ConsolidationConfig,
    ContextConsolidator,
    ContextSummaryGenerator,
)
from agent_harness.context import ContextBuilder
from agent_harness.context_governor import ContextGovernor, ContextGovernorConfig
from agent_harness.memory import Dream, LocalMemoryStore
from agent_harness.providers.openai_compatible import OpenAICompatibleProvider
from agent_harness.runner import AgentRunner
from agent_harness.runtime import AgentRuntime, RuntimeResult
from agent_harness.runtime_io import (
    OutputSegmentEnded,
    OutputTextDelta,
    RuntimeRequest,
    RuntimeStreamEvent,
    RuntimeStreamHandler,
)
from agent_harness.skills import SkillCatalog
from agent_harness.status_builder import RuntimeStatusBuilder
from agent_harness.storage import JsonlSessionStore
from agent_harness.token_estimation import build_default_token_estimator
from agent_harness.tools import (
    ActivateSkillTool,
    BashTool,
    CurrentTimeTool,
    EditTool,
    FindTool,
    GrepTool,
    LsTool,
    ReadArtifactTool,
    ReadSkillResourceTool,
    ReadTool,
    ToolExecutionPolicy,
    ToolRegistry,
    WriteTool,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools when they are useful."
)
DEFAULT_SESSION_ID = "cli:default"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DAO Agent Harness in a terminal.")
    parser.add_argument("--model", default=os.getenv("AGENT_HARNESS_MODEL"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("AGENT_HARNESS_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("AGENT_HARNESS_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument(
        "--max-injected-inputs-per-run",
        type=int,
        default=int(os.getenv("AGENT_HARNESS_MAX_INJECTED_INPUTS_PER_RUN", "5")),
        help="Maximum follow-up PendingInput values absorbed by one Runner.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--tool-timeout", type=float, default=60.0)
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=os.getenv("AGENT_HARNESS_CONTEXT_WINDOW_TOKENS"),
        help="Enable ContextGovernor with this model context-window size.",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument(
        "--proactive-input-reserve-tokens",
        type=int,
        default=int(
            os.getenv("AGENT_HARNESS_PROACTIVE_INPUT_RESERVE_TOKENS", "2048")
        ),
        help="Input tokens reserved by the proactive post-SAVE consolidation probe.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=os.getenv("AGENT_HARNESS_MAX_INPUT_TOKENS"),
        help=(
            "Maximum tokens accepted from one user input. Defaults to half of the "
            "available input budget when context-window governance is enabled."
        ),
    )
    parser.add_argument("--max-tool-result-chars", type=int, default=16_000)
    parser.add_argument(
        "--session-id",
        default=os.getenv("AGENT_HARNESS_SESSION_ID", DEFAULT_SESSION_ID),
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path(
            os.getenv(
                "AGENT_HARNESS_SESSION_DIR",
                str(Path.home() / ".dao-agent" / "sessions"),
            )
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(
            os.getenv(
                "AGENT_HARNESS_ARTIFACT_DIR",
                str(Path.home() / ".dao-agent" / "artifacts"),
            )
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            os.getenv(
                "AGENT_HARNESS_CHECKPOINT_DIR",
                str(Path.home() / ".dao-agent" / "checkpoints"),
            )
        ),
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path(
            os.getenv(
                "AGENT_HARNESS_MEMORY_DIR",
                str(Path.home() / ".dao-agent" / "memory"),
            )
        ),
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    return parser


class CliStreamRenderer:
    """Render ephemeral Runtime stream segments without owning conversation state."""

    def __init__(self) -> None:
        self.received_delta = False
        self._active_segment: int | None = None
        self._segment_has_text = False
        self._needs_prefix = False

    def start(self) -> None:
        print("agent> ", end="", flush=True)

    def handle(self, event: RuntimeStreamEvent) -> None:
        if isinstance(event, OutputTextDelta):
            if self._active_segment != event.segment_index:
                if self._needs_prefix:
                    print("agent> ", end="", flush=True)
                self._active_segment = event.segment_index
                self._segment_has_text = False
                self._needs_prefix = False
            self.received_delta = True
            self._segment_has_text = True
            print(event.text, end="", flush=True)
            return

        if isinstance(event, OutputSegmentEnded):
            if self._segment_has_text:
                print()
                self._needs_prefix = True
            self._active_segment = event.segment_index
            self._segment_has_text = False
            return

        raise TypeError(f"Unsupported Runtime stream event: {type(event).__name__}")

    def finish(self, result: RuntimeResult) -> None:
        if result.status in {"completed", "limit_reached"}:
            if not self.received_delta:
                print(result.final_content or "")
            if result.has_pending_continuation:
                print(
                    "agent> Continuing with "
                    f"{result.remaining_pending_count} queued message(s)."
                )
            return


        if result.status == "injected":
            print("Message joined the active agent run.")
            return

        if result.status == "paused":
            print("Run paused for message revision.")
            if result.side_effect_status != "none":
                print(
                    "Warning: discarded tool calls may have external effects: "
                    + ", ".join(result.discarded_tool_call_ids)
                )
            return

        detail = result.error or result.stop_reason
        if self.received_delta:
            print("agent> ", end="")
        print(f"Run {result.status}: {detail}")

async def run_chat(args: argparse.Namespace) -> int:
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_s=args.timeout,
    )
    artifact_policy = ArtifactPolicy()
    artifact_store = LocalArtifactStore(args.artifact_dir)
    skill_catalog = SkillCatalog.for_workspace(args.workspace)
    tools = ToolRegistry(
        ToolExecutionPolicy(default_timeout_s=args.tool_timeout),
        artifact_store=artifact_store,
        artifact_policy=artifact_policy,
    )
    tools.register(CurrentTimeTool())
    tools.register(ReadTool(args.workspace))
    tools.register(LsTool(args.workspace))
    tools.register(FindTool(args.workspace))
    tools.register(GrepTool(args.workspace))
    tools.register(WriteTool(args.workspace))
    tools.register(EditTool(args.workspace))
    tools.register(BashTool(args.workspace))
    tools.register(ReadArtifactTool(artifact_store, artifact_policy))
    tools.register(ActivateSkillTool(skill_catalog))
    tools.register(ReadSkillResourceTool(skill_catalog))
    store = JsonlSessionStore(args.session_dir)
    checkpoint_store = JsonFileCheckpointStore(args.checkpoint_dir)
    memory_store = LocalMemoryStore(
        getattr(args, "memory_dir", Path(args.session_dir).parent / "memory")
    )
    context_governor = None
    consolidator = None
    token_estimator = None
    dream = None
    if args.context_window_tokens is not None or args.max_input_tokens is not None:
        token_estimator = build_default_token_estimator(provider)
    if args.context_window_tokens is not None:
        context_governor = ContextGovernor(
            ContextGovernorConfig(
                context_window_tokens=args.context_window_tokens,
                max_completion_tokens=args.max_completion_tokens,
                max_tool_result_chars=args.max_tool_result_chars,
            ),
            token_estimator=token_estimator,
        )
        consolidator = ContextConsolidator(
            generator=ContextSummaryGenerator(provider, model=args.model),
            token_estimator=token_estimator,
            session_store=store,
            memory_store=memory_store,
            model=args.model,
            config=ConsolidationConfig(
                context_window_tokens=args.context_window_tokens,
                max_completion_tokens=args.max_completion_tokens,
                proactive_input_reserve_tokens=args.proactive_input_reserve_tokens,
            ),
        )
        dream = Dream(store=memory_store, provider=provider, model=args.model)
    max_input_tokens = args.max_input_tokens
    if max_input_tokens is None and args.context_window_tokens is not None:
        input_budget = (
            args.context_window_tokens - args.max_completion_tokens - 1024
        )
        max_input_tokens = max(1, input_budget // 2)
    runtime = AgentRuntime(
        runner=AgentRunner(
            provider,
            context_governor=context_governor,
            status_builder=RuntimeStatusBuilder(),
        ),
        session_store=store,
        context_builder=ContextBuilder(
            args.workspace,
            skill_catalog=skill_catalog,
            memory_store=memory_store,
        ),
        tools=tools,
        model=args.model,
        max_turns=args.max_turns,
        extra_system_sections=(args.system_prompt,),
        consolidator=consolidator,
        checkpoint_store=checkpoint_store,
        max_injected_inputs_per_run=args.max_injected_inputs_per_run,
        max_input_tokens=max_input_tokens,
        input_token_estimator=token_estimator,
        dream=dream,
    )

    print(
        "DAO Agent Harness ready. Commands: "
        "/pause, /edit <text>, /resume, /retry, /clear, /exit"
    )
    execution_task: asyncio.Task[RuntimeResult] | None = None
    renderer: CliStreamRenderer | None = None
    input_task: asyncio.Task[str] | None = None
    paused_target_id: str | None = None

    def start_execution(
        factory: Callable[
            [RuntimeStreamHandler],
            Coroutine[object, object, RuntimeResult],
        ],
    ) -> None:
        nonlocal execution_task, renderer
        renderer = CliStreamRenderer()
        renderer.start()
        execution_task = asyncio.create_task(factory(renderer.handle))

    async def stop_active_for_revision() -> RuntimeResult:
        nonlocal execution_task, renderer, paused_target_id
        paused = await runtime.pause_for_revision(args.session_id)
        if execution_task is not None:
            await asyncio.gather(execution_task, return_exceptions=True)
        execution_task = None
        if renderer is not None:
            renderer.finish(paused)
        renderer = None
        paused_target_id = paused.revision_target_input_id
        return paused

    while True:
        if input_task is None:
            prompt = "edit> " if paused_target_id is not None else "you> "
            input_task = asyncio.create_task(asyncio.to_thread(input, prompt))
        waiters: set[asyncio.Task[object]] = {input_task}
        if execution_task is not None:
            waiters.add(execution_task)
        done, _pending = await asyncio.wait(
            waiters,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if execution_task is not None and execution_task in done:
            completed_task = execution_task
            execution_task = None
            try:
                result = completed_task.result()
            except Exception as exc:
                print()
                print(
                    "Run failed before the response could be committed; "
                    "the pending input was retained when possible: "
                    f"{type(exc).__name__}: {exc}"
                )
                renderer = None
            else:
                if renderer is not None:
                    renderer.finish(result)
                renderer = None
                if result.status == "paused":
                    paused_target_id = result.revision_target_input_id
                elif result.has_pending_continuation:
                    start_execution(
                        lambda on_stream: runtime.run_next(
                            args.session_id,
                            on_stream=on_stream,
                        )
                    )

        if input_task not in done:
            continue
        try:
            text = input_task.result().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            if execution_task is not None:
                await stop_active_for_revision()
            await runtime.wait_for_background_tasks()
            return 0
        finally:
            input_task = None

        if not text:
            continue
        if text in {"/exit", "/quit"}:
            if execution_task is not None:
                try:
                    result = await execution_task
                except Exception as exc:
                    print(
                        "Run failed before exit: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    if renderer is not None:
                        renderer.finish(result)
                execution_task = None
                renderer = None
            await runtime.wait_for_background_tasks()
            return 0
        if text == "/pause":
            paused = await stop_active_for_revision()
            if paused.status == "idle":
                print("No pending run to pause.")
            elif paused.revision_target_input_id is not None:
                print("Enter replacement text, or use /resume without editing.")
            continue
        if text == "/clear":
            if execution_task is not None:
                await stop_active_for_revision()
            await runtime.cancel_background_consolidation(args.session_id)
            runtime.discard_paused_run(args.session_id)
            paused_target_id = None
            store.delete(args.session_id)
            checkpoint_store.delete(args.session_id)
            print("Conversation cleared.")
            continue
        if text == "/resume":
            if paused_target_id is None:
                print("No paused run to resume.")
                continue
            paused_target_id = None
            start_execution(
                lambda on_stream: runtime.restart_pending(
                    args.session_id,
                    on_stream=on_stream,
                )
            )
            continue
        if text.startswith("/edit "):
            if paused_target_id is None:
                print("Pause a run before editing its PendingInput.")
                continue
            replacement = text.removeprefix("/edit ").strip()
            if not replacement:
                print("Replacement text must be non-empty.")
                continue
            target_id = paused_target_id
            paused_target_id = None
            runtime.revise_paused_input(args.session_id, target_id, replacement)
            start_execution(
                lambda on_stream: runtime.restart_pending(
                    args.session_id,
                    on_stream=on_stream,
                )
            )
            continue
        if paused_target_id is not None:
            target_id = paused_target_id
            paused_target_id = None
            runtime.revise_paused_input(args.session_id, target_id, text)
            start_execution(
                lambda on_stream: runtime.restart_pending(
                    args.session_id,
                    on_stream=on_stream,
                )
            )
            continue
        if text == "/retry":
            if execution_task is not None:
                print("A run is already active.")
                continue
            start_execution(
                lambda on_stream: runtime.run_next(
                    args.session_id,
                    on_stream=on_stream,
                )
            )
            continue
        if execution_task is not None:
            queued = runtime.enqueue_input(args.session_id, uuid4().hex, text)
            print(f"Message queued for injection: {queued.id}")
            continue
        request = RuntimeRequest(
            session_id=args.session_id,
            source_message_id=uuid4().hex,
            content=text,
        )
        start_execution(
            lambda on_stream: runtime.submit(
                request,
                on_stream=on_stream,
            )
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model or AGENT_HARNESS_MODEL is required")
    if args.max_turns <= 0:
        parser.error("--max-turns must be greater than zero")
    if args.max_injected_inputs_per_run < 0:
        parser.error("--max-injected-inputs-per-run must be non-negative")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.tool_timeout <= 0:
        parser.error("--tool-timeout must be greater than zero")
    if args.context_window_tokens is not None and args.context_window_tokens <= 0:
        parser.error("--context-window-tokens must be greater than zero")
    if args.max_completion_tokens < 0:
        parser.error("--max-completion-tokens must be non-negative")
    if args.proactive_input_reserve_tokens < 0:
        parser.error("--proactive-input-reserve-tokens must be non-negative")
    if args.max_input_tokens is not None and args.max_input_tokens <= 0:
        parser.error("--max-input-tokens must be greater than zero")
    if args.max_tool_result_chars < 0:
        parser.error("--max-tool-result-chars must be non-negative")
    if 0 < args.max_tool_result_chars < 32:
        parser.error("--max-tool-result-chars must be zero or at least 32")
    if (
        args.context_window_tokens is not None
        and args.context_window_tokens <= args.max_completion_tokens + 1024
    ):
        parser.error(
            "--context-window-tokens must exceed "
            "--max-completion-tokens plus the 1024-token safety buffer"
        )
    if not args.session_id.strip():
        parser.error("--session-id must be non-empty")
    return asyncio.run(run_chat(args))


if __name__ == "__main__":
    raise SystemExit(main())
