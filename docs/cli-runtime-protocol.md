# CLI 与 Runtime 对外协议

> 状态：Implemented v0.2  
> 最后更新：2026-08-11  
> 范围：DAO Agent Harness 第一版 CLI Adapter 与 AgentRuntime 的交互边界

## 1. 产品范围

DAO v1 采用 **CLI-first** 路线，第一版只实现 CLI Adapter，不实现 GUI、HTTP API、WebSocket、
Channel 或通用消息总线。

CLI 是 AgentRuntime 的外部适配器，不拥有对话历史，也不直接调用 AgentRunner。Session、
PendingInput、上下文构建、模型—工具循环与最终保存全部由 Runtime 主链路负责。

```text
stdin
  → CLI Adapter
  → RuntimeRequest
  → AgentRuntime
  → RuntimeStreamEvent（临时展示）
  → RuntimeResult（正式结果）
  → stdout
```

通用 `InboundMessage`、`OutboundMessage` 和 GUI 控件协议不进入 v1；未来新增应用入口时，在
Runtime 协议外增加 Adapter，而不修改 Runner 核心。

## 2. 输入协议

```python
@dataclass(frozen=True, slots=True)
class RuntimeRequest:
    session_id: str
    source_message_id: str
    content: str
```

- `session_id` 定位 Session；
- `source_message_id` 是 CLI 为一次用户输入生成的稳定外部 ID；
- `content` 第一版只接受非空纯文本；
- Runtime 先将其持久化为 PendingInput，再启动 Agent Loop；
- RuntimeRequest 不携带 Provider、模型、工具、system prompt 或持久化 `run_id`；
- `/exit`、`/clear` 等 CLI 命令由 Adapter 解释，不作为用户消息提交。

## 3. 流式输出协议

流式语义参考 nanobot 的分段模型：一次模型文本流结束后，如果还要执行工具或继续生成，结束
事件携带 `resuming=True`；真正的候选最终回答结束时携带 `resuming=False`。

```python
@dataclass(frozen=True, slots=True)
class OutputTextDelta:
    input_id: str
    segment_index: int
    text: str


@dataclass(frozen=True, slots=True)
class OutputSegmentEnded:
    input_id: str
    segment_index: int
    resuming: bool


RuntimeStreamEvent = OutputTextDelta | OutputSegmentEnded
RuntimeStreamHandler = Callable[
    [RuntimeStreamEvent],
    Awaitable[None] | None,
]
```

`OutputTextDelta` 与 Provider 层 `TextDelta` 是不同协议。Provider 事件只描述厂商响应；Runner
完成归一化后，Runtime 才向 Adapter 暴露 `RuntimeStreamEvent`。

事件约束：

- 流事件只用于临时展示，不持久化，不进入 Session Entry Tree；
- `input_id` 使用 PendingInput ID，不创建额外 run ID；
- `segment_index` 从 0 开始，每结束一个模型输出段后递增；
- 工具调用前发出 `OutputSegmentEnded(resuming=True)`；
- 候选最终回答完成后发出 `OutputSegmentEnded(resuming=False)`；
- Provider 不支持流式调用时，不产生流事件，直接返回 RuntimeResult；
- 第一版不暴露 reasoning、工具进度、审批、诊断或 Timeline 事件。

典型工具循环为：

```text
OutputTextDelta*
→ OutputSegmentEnded(resuming=True)
→ execute tools
→ OutputTextDelta*
→ OutputSegmentEnded(resuming=False)
→ SAVE
→ RuntimeResult
```

## 4. 正式结果协议

现有 RuntimeResult 保持为一次 Runtime 执行的终态结果：

```python
@dataclass(frozen=True, slots=True)
class RuntimeResult:
    session_id: str
    input_id: str | None
    status: RuntimeStatus
    final_content: str | None = None
    stop_reason: str | None = None
    tools_used: tuple[str, ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    error: str | None = None
    revision_target_input_id: str | None = None
    discarded_message_count: int = 0
    side_effect_status: SideEffectStatus = "none"
    discarded_tool_call_ids: tuple[str, ...] = ()
```

流事件表示“模型正在生成什么”，RuntimeResult 表示“Runtime 最终确认了什么”。

- `completed` 和 `limit_reached` 只能在 SAVE 成功后返回；
- `failed` 和 `cancelled` 不提交工作消息，PendingInput 保留；
- SAVE 异常直接向 Adapter 传播，不生成虚假的成功结果；
- 即使候选回答已经流式显示，SAVE 失败时也不得将其视为已提交历史。

这与 nanobot 的主时序一致：流式分段发生在 RUN 内，正式外部响应发生在 SAVE 之后。

## 5. Runtime 公开接口

```python
class AgentRuntime:
    async def submit(
        self,
        request: RuntimeRequest,
        *,
        on_stream: RuntimeStreamHandler | None = None,
    ) -> RuntimeResult: ...

    async def run_next(
        self,
        session_id: str,
        *,
        on_stream: RuntimeStreamHandler | None = None,
    ) -> RuntimeResult: ...

    async def pause_for_revision(
        self, session_id: str, input_id: str | None = None,
    ) -> RuntimeResult: ...

    def revise_paused_input(
        self, session_id: str, input_id: str, content: str,
    ) -> PendingInput: ...

    async def restart_pending(self, session_id: str) -> RuntimeResult: ...
```

- `submit()` 先持久化请求，再执行当前队首；
- `run_next()` 用于执行或重试已经存在的队首 PendingInput；
- 是否传入 `on_stream` 决定本次调用是否请求流式输出；
- `enqueue_input()` 继续作为低层队列接口，但 CLI 正常对话不直接调用；
- v1 不引入持久化 ExecutionHandle 或运行标识；主动暂停通过 Session 范围的内存 ActiveRun 完成。

## 6. CLI 渲染规则

- 收到 OutputTextDelta 时立即追加到 stdout；
- 收到 `OutputSegmentEnded(resuming=True)` 时结束当前显示段并等待下一段；
- 收到 `OutputSegmentEnded(resuming=False)` 时只表示候选回答生成结束；
- 成功 RuntimeResult 到达后确认本轮完成，不重复打印已经流式展示的全文；
- 如果没有收到任何文本增量，则打印 `RuntimeResult.final_content`；
- RuntimeResult 失败时显示错误并允许用户重试现有 PendingInput；
- SAVE 抛出异常时明确提示回答未提交，且不得自行修改 Session；
- CLI 不维护本地 history，后续对话始终由同一 Session 继续。

## 7. 主动暂停与追加输入

CLI 在 AgentRuntime 运行期间继续读取输入。普通文本先保存为 PendingInput，并由当前 Runner 在
安全点吸收；`/pause` 取消当前执行并默认选择最新追加消息作为修订目标。暂停后直接输入替换文本，
或使用 `/edit <text>`，会持久化新 revision 并从截断前缀重新运行；`/resume` 保留原内容续跑。

详细的截断规则、paused latch 和工具副作用提示见
[主动暂停与消息修订设计](pause-revision-design.md)。

## 8. 暂缓事项

- GUI、HTTP、WebSocket 和聊天 Channel Adapter；
- 通用 InboundMessage / OutboundMessage；
- reasoning、工具执行和审批等富事件；
- 持久化 ExecutionHandle 和跨进程暂停快照；
- Runtime Timeline、diagnostics 和事件持久化；
- 富文本、多模态和 Artifact 展示协议。

## 9. 当前实施状态

以下能力已经实现并有测试覆盖：

- RuntimeRequest、OutputTextDelta、OutputSegmentEnded 与 RuntimeStreamHandler；
- Runner 在工具调用前发送 `resuming=True`，在最终候选回答后发送 `resuming=False`；
- max_turns 框架终止回答作为最终流式分段输出；
- Provider 不支持 stream 时回退到完整响应且不产生伪流事件；
- CLI 通过 JsonlSessionStore 和 AgentRuntime 完成多轮对话，不再维护本地 history；
- CLI 通过 LocalArtifactStore 外置大结果，并自动注册 `read_artifact` 分页读取工具；
- `--artifact-dir` 可覆盖默认的 `~/.dao-agent/artifacts`；
- `/retry` 重新执行保留的 PendingInput，`/clear` 删除当前 CLI Session；
- 运行中追加、`/pause`、`/edit <text>`、直接替换文本与 `/resume`；
- 两轮 Session 上下文、流事件映射、失败保留与 CLI 重试端到端测试。
