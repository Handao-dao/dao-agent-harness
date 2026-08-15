# Runner 设计文档

> 状态：Implemented v0.8  
> 最后更新：2026-08-14  
> 范围：当前模型—工具循环与已接受的临时上下文治理边界

## 1. 定位

AgentRunner 是无 Session 状态的内部循环：接收完整工作消息，调用 Provider，执行工具，并
返回追加后的完整工作消息。它以 nanobot 的已验证循环行为为参考，但不复制 Channel、Memory、
Subagent、checkpoint 等外围能力。

```text
AgentMessage[]
  → ContextBuilder 消息投影
  → Provider
  → LLMResponse
  → AssistantMessage / ToolResultMessage
  → 下一轮或终止
```

## 2. 与 Provider 的边界

Provider 完成厂商协议相关工作：

- 请求和响应格式转换；
- SSE 或其他流式协议解析；
- 聚合 tool call ID、名称和 arguments 分片；
- 返回 Provider-neutral `LLMResponse` 或统一流事件。

Runner 完成 Harness 领域加工：

- 把 `LLMResponse` 转为 `AssistantMessage`；
- 把 `ToolCallRequest` 转为 `ToolCall`；
- 执行工具并生成 `ToolResultMessage`；
- 累计 usage，判断停止条件并构造 `AgentRunResult`。

只有 ContextBuilder 产生的 `Mapping` 消息会进入 Provider；dict 消息不进入 Runner 结果或
Session。

## 3. 公开协议

```python
@dataclass(frozen=True, slots=True)
class AgentRunSpec:
    initial_messages: Sequence[AgentMessage]
    tools: ToolRegistry
    model: str
    system_prompt: str | None = None
    max_turns: int = 20
    stream: bool = False
    on_text_delta: TextDeltaHandler | None = None
    on_stream_end: StreamEndHandler | None = None
    model_message_start: int = 0
    current_turn_start: int | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    final_content: str | None
    messages: tuple[AgentMessage, ...]
    status: RunStatus
    stop_reason: str
    tools_used: tuple[str, ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    error: str | None = None
```

Runner 复制 `initial_messages` 的容器后只追加，不修改输入序列或已有消息。AgentMessage 为
不可变 dataclass，无需深拷贝每条消息。

## 4. 主循环

每个 turn：

1. ContextGovernor 生成本次合法、预算内的强类型模型视图；
2. 可选 RuntimeStatusBuilder 生成状态并追加到 working messages；
3. ContextBuilder 把治理视图与最新状态投影为模型消息；
4. Runner 调用 Provider；
5. Provider 失败则返回 `failed/provider_error`；
6. 若模型请求执行工具，提交 AssistantMessage，划分并执行工具批次，再提交 ToolResultMessage；
7. 若模型不给工具调用，提交最终 AssistantMessage并返回 `completed/model_stop`；
8. 达到 max_turns 时追加框架终止 AssistantMessage，返回
   `limit_reached/max_turns`。

工具执行参考 nanobot 的批次划分：连续的 `parallel_safe` 调用组成并行批次，`sequential`
调用形成单独的执行屏障，屏障前后的并行批次互不穿越。并行结果按真实完成顺序追加，并通过
`tool_call_id` 关联，不依赖调用位置。

## 5. 工具调用判断

`LLMResponse.should_execute_tools` 不能只检查 tool_calls 是否非空。只有 finish_reason 属于：

```text
tool_calls | function_call | stop
```

并且存在完整 tool call 时才执行。这保留 OpenAI-compatible 的宽容行为，同时避免错误、拒绝
或长度截断响应里残留的 tool_calls 被误执行。

ToolRegistry 当前在执行前完成工具查找、schema 驱动的安全参数转换和常用 JSON Schema 子集
校验。Execution Policy v0.2 已把单次执行、timeout 和普通结果归一化一并收归 Registry，
Runner 只保留批次与并发调度。工具调用
和结果通过 `tool_call_id` 关联，不依赖结果排列位置。工具找不到、参数无效、工具主动返回
`Error` 或普通执行异常时，Runner 生成 `ToolResultMessage(is_error=True)`，而不是让整次 Run
失败。`asyncio.CancelledError` 不包装成普通工具错误。完整规则见
[工具注册、校验与执行设计](tool-execution-design.md)。

## 6. 流式响应

Runner 只认识三种 Provider-neutral 事件：

- `TextDelta`；
- `ToolCallCompleted`；
- `ResponseCompleted`。

Provider 没有 `stream()` 时自动回退到 `complete()`。流必须恰好包含一个
ResponseCompleted；缺少或重复都作为 provider_error。Runner 仅在完整
ToolCallCompleted 到达后执行工具。

## 7. 终止和保存语义

```python
RunStatus = Literal["completed", "failed", "cancelled", "limit_reached"]
```

| 状态 | Runner 消息尾部 | Runtime SAVE |
|---|---|---|
| completed | 完整最终 AssistantMessage | 提交 |
| limit_reached | 框架终止 AssistantMessage | 提交 |
| failed | 不追加伪造错误回答 | 不提交 |
| cancelled | 保留调用前工作消息 | 不提交 |

正常但内容为空的模型响应使用稳定的框架兜底文本，以满足 AssistantMessage 和历史协议。达到
循环上限同样使用明确文本闭合历史，而不是返回以 ToolResult 结尾的半完成对话。

SAVE 属于 AgentRuntime，Runner 不读取或写入 SessionStore。

## 8. 不变量

1. Runner 不直接访问 Session、Channel、MessageBus 或 TransportMessage；
2. 输入前缀不修改，运行消息只追加；
3. Runner 和 Session 只使用 AgentMessage；
4. Provider 消息只存在于调用边界；
5. 每个已执行 ToolCall 都生成一个同 ID 的 ToolResultMessage；
6. 普通工具错误不终止循环；
7. 失败结果不伪造 AssistantMessage；
8. completed 与 limit_reached 的消息尾部都可安全保存。
9. 启用状态栏时，每次真实模型决策前只追加一个 RuntimeStatusMessage。

## 9. 已完成的横向能力与暂缓项

- checkpoint：已采用 `awaiting_tools`、`tools_completed`、`final_response` 三个节点，见
  [ContextCheckpoint 设计](checkpoint-design.md)；
- 消息插入：已在工具批次结束和候选最终回答结束后检查 PendingInput 新增后缀，单 Runner 最多
  吸收五条，见 [消息插入设计](message-injection-design.md)；
- RunEvent、StreamEvent 之外的追踪事件和诊断事件；
- ContextGovernor 已实现；Memory 仍暂缓；
- Runtime Status 基础框架已实现；字段和生命周期见
  [Runtime Status 设计](runtime-status-design.md)；
- Provider retry 策略；
- 工具 timeout 已实现；approval、幂等和外部副作用恢复仍暂缓；
- 多模态和 reasoning blocks。

这些能力不改变当前 Runner 的领域输入输出边界，应作为独立组件逐步加入。

ContextGovernor 的强类型协议、三级治理顺序、任务锚点和 context_limit 失败语义见
[Runner ContextGovernor 设计](context-governor-design.md)。

## 10. 当前优化范围：稳定性收口

Runner 仍只优化已有模型—工具主链路，不引入 Memory、MCP 或通用结果适配器。Skill 通过标准
ToolRegistry、ToolOutput metadata 和 ContextBuilder 前缀接入，不增加 Runner 专用状态机；目标仍是
让相同输入具有确定、可恢复且可测试的行为，重点包括：

- ContextGovernor 产生的请求视图与 Provider 实际请求保持一致；
- 不改变 working messages、Session SAVE 边界和任务锚点语义；
- 减少无变化上下文的重复投影和 token 估算；
- 明确 Provider、治理、工具、Checkpoint 和消息插入各自的失败边界；
- 保证工具循环、候选最终回答、消息插入和 `max_turns` 在边界条件下稳定终止；
- 只参考 nanobot 中适合上述问题的成熟实现，不追求源码结构或行为兼容。

工具结果的语义级摘要、结构化视图和按工具特性适配已记录为后续 Tool Runtime 方向；当前
ContextGovernor 继续承担通用首尾裁剪、历史 micro-compaction 和硬预算兜底。
