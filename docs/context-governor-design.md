# Runner ContextGovernor 设计

> 状态：Implemented v0.3  
> 最后更新：2026-08-14  
> 范围：每次 Provider 调用前的临时工具链修复、工具结果治理与 emergency snip

## 1. 定位

ContextGovernor 是 AgentRunner 内每次调用 Provider 前执行的临时上下文治理组件。它参考
nanobot 已验证的执行顺序，但适配 DAO 的强类型 AgentMessage、独立 system prompt、Entry Tree
和耐久 ContextSummary。

```text
完整 working AgentMessage（SAVE 使用）
        │
        ├──────────────────────────────→ AgentRunResult → Runtime SAVE
        │
        └→ model_message_start 之后的模型可见切片
             → ContextGovernor.prepare()
             → 临时 GovernedContext
             → RuntimeStatusBuilder（可选）
             → ContextBuilder.build_messages()
             → Provider
```

ContextGovernor 不修改 Runner 的 working messages，不写 Session，不生成 ContextSummary。其所有
修复、替换和裁剪只影响当前这一次 Provider 请求。

## 2. 与 Consolidator 的边界

| 组件 | 作用域 | 是否调用模型 | 是否持久化 | 主要目的 |
|---|---|---:|---:|---|
| ContextConsolidator | 跨 Turn 历史 | 是 | 是 | 生成结构化 ContextSummary |
| ContextGovernor | 单次 Provider 请求 | 否 | 否 | 保证请求合法并控制即时上下文体积 |

Consolidator 在 Runtime SAVE 后后台处理已经提交的历史，并在下一轮 PREPARE 使用真实输入复检；
ContextGovernor 在 Runner 的每轮模型调用前处理当前 working messages，因此可以治理本轮新产生的
大型工具结果。

## 3. 公开协议

```python
@dataclass(frozen=True, slots=True)
class ContextGovernorConfig:
    context_window_tokens: int
    max_completion_tokens: int = 4096
    safety_buffer_tokens: int = 1024
    max_tool_result_chars: int = 16_000
    microcompact_keep_recent: int = 10
    microcompact_min_chars: int = 500
    compactable_tool_names: frozenset[str] = frozenset()

    @property
    def input_budget(self) -> int:
        return (
            self.context_window_tokens
            - self.max_completion_tokens
            - self.safety_buffer_tokens
        )
```

```python
@dataclass(frozen=True, slots=True)
class ContextGovernanceReport:
    estimated_tokens_before: int | None
    estimated_tokens_after: int | None
    orphan_results_dropped: int = 0
    missing_results_backfilled: int = 0
    tool_results_compacted: int = 0
    tool_results_truncated: int = 0
    history_messages_snipped: int = 0
    active_turn_messages_snipped: int = 0
    runtime_statuses_dropped: int = 0
    skill_instructions_deduplicated: int = 0
    estimation_source: str | None = None
```

```python
@dataclass(frozen=True, slots=True)
class GovernedContext:
    messages: tuple[AgentMessage, ...]
    report: ContextGovernanceReport
```

```python
class ContextGovernor:
    async def prepare(
        self,
        *,
        messages: Sequence[AgentMessage],
        current_turn_start: int,
        model: str,
        system_prompt: str | None,
        tools: Sequence[Mapping[str, Any]],
    ) -> GovernedContext: ...
```

传给 Governor 的 `messages` 已经是 `model_message_start` 之后的模型可见切片，已被耐久
ContextSummary 覆盖的原始历史不会重新进入治理输入。`current_turn_start` 是该切片内的相对索引，
指向本次 PendingInput 转换出的 UserMessage。Runtime 在构造 AgentRunSpec 时记录 working messages
中的绝对索引，Runner 切片后换算为相对索引。该位置用于识别本轮初始任务锚点和统计历史/活跃
轨迹，不再表示其后的整个当前 Turn 都不可裁剪。

Report 是本次请求的非持久化派生信息，为测试和未来 Timeline 提供数据；它不是诊断事件，也不
进入 Session。

## 4. 三级治理顺序

ContextGovernor 的常态职责是协议保障，而不是主动压缩所有上下文。只有前一层仍超出输入预算时
才进入下一层：

```text
NORMAL
  drop orphan tool results
  backfill missing tool results
  truncate oversized tool results
  estimate complete Provider request
       ├─ within budget → return
       ▼
PRESSURE
  drop expired runtime-status messages
  micro-compact eligible historical tool results
  estimate again when the view changed
       ├─ within budget → return
       ▼
EMERGENCY
  keep task anchors + most recent legal message tail
  repair tool chains again
  estimate and enforce the hard limit
```

无修复或裁剪发生时，NORMAL 复用第一次估算，不为 Report 重复计算同一请求。Emergency snip
可能跨过历史或当前 Turn 的消息，因此完成后必须再次执行结构修复。

RuntimeStatusBuilder 在 Governor 返回后才生成本轮最新状态，因此 PRESSURE 删除的只可能是旧状态。
删除数量进入 Report，并由最新状态的 `context_visibility` 告知模型。

## 5. 工具链修复

### 5.1 孤立 ToolResult

如果 ToolResultMessage 的 `tool_call_id` 在它之前没有匹配的 AssistantMessage.tool_calls，临时
Provider 上下文删除该结果。正式 working messages 不变。

### 5.2 缺失 ToolResult

如果 AssistantMessage 声明了 ToolCall，但后续没有同 ID 的 ToolResultMessage，在该 Assistant
消息之后插入临时错误结果：

```text
[Tool result unavailable — call was interrupted or lost]
```

临时结果沿用原 tool call ID 和工具名称，`is_error=True`。该占位只修复模型协议，不证明工具
没有执行，也不得写入 Session。未来 checkpoint 恢复仍根据工具副作用状态单独判断。

## 6. Micro-compaction

进入 PRESSURE 后先删除旧 RuntimeStatusMessage 并重新估算；若已经回到预算内，则不再损失工具
结果。仍超限时才执行历史工具结果 micro-compaction。状态删除和工具压缩都只作用于单次模型
视图，Session 中的原消息不变。

不是所有工具结果都可以安全省略。第一版由配置显式提供 `compactable_tool_names`，默认空集合；
未来可迁移为 ToolRegistry 中的工具元数据。

Micro-compaction 只在 NORMAL 仍然超出预算后执行，并且只处理 `current_turn_start` 之前的历史
结果。当前 Turn 的工具结果不整条替换为 omission 占位。对于允许压缩的历史结果：

- 保留最近 `microcompact_keep_recent` 个结果；
- 只处理长度达到 `microcompact_min_chars` 的旧结果；
- 将内容替换为稳定占位：`[<tool_name> result omitted from model context]`；
- ToolResultMessage 的 tool_call_id、tool_name 和 is_error 关系保持不变。

这里不调用模型，不生成自由文本摘要，也不压缩 UserMessage 或 AssistantMessage。预算充足时不
执行 micro-compaction，避免无必要的信息损失。

## 7. 单个工具结果预算

所有仍然过大的 ToolResultMessage 都受 `max_tool_result_chars` 约束。裁剪采用首尾保留，而不是
nanobot 当前只保留前缀的方式：

```text
<head>

... [tool result truncated for model context: N chars omitted] ...

<tail>
```

这可以同时保留命令开头、错误栈结尾或文件尾部结论。原始 ToolResultMessage 不变。对于新产生的
大型成功结果，ToolRegistry 已优先通过 ArtifactStore 外置原文；ContextGovernor 仍负责旧历史、
未配置 ArtifactStore 的嵌入式运行和最终请求预算。

`max_tool_result_chars <= 0` 表示禁用字符级裁剪，不表示删除全部内容。
启用时至少为 32 个字符，避免裁剪标记本身吞掉全部首尾预览。

## 8. Token 预算与估算

输入预算沿用 nanobot 和当前 Consolidator 已采用的公式：

```text
input_budget
= context_window_tokens
- max_completion_tokens
- safety_buffer_tokens
```

估算必须覆盖完整 Provider 请求：system prompt、治理后的消息以及 Tool definitions。优先使用
Provider counter，其次使用可选 tiktoken，最后允许使用明确标注来源的字符启发式估算。响应中的
usage 只能用于事后观测，不能替代发送前估算。

字符估算是 emergency snip 的保守兜底，不改变 ContextConsolidator 当前“估算不可用则跳过耐久
压缩”的失败语义。

NORMAL 中如果协议修复和单结果裁剪都没有改变模型视图，`estimated_tokens_after` 直接复用首次
估算；只有视图发生变化、进入 PRESSURE 或进入 EMERGENCY 时才重新估算。

## 9. Emergency snip

只有 NORMAL 和 PRESSURE 都无法满足 input_budget 时才执行。它是健康循环之外的最后兜底，不是
日常压缩手段。规则如下：

1. system prompt 和 Tool definitions 不可裁剪；
2. `current_turn_start` 指向的初始 UserMessage 是任务锚点，必须保留；
3. 当前 Turn 最新 UserMessage 如果不同于初始锚点，也必须保留；
4. 最新完整消息块必须保留，Assistant tool calls 与其连续 ToolResult 作为一个裁剪区间；
5. 其余历史与当前 Turn 中间执行轨迹都可以从临时模型视图删除；
6. 历史侧尾部只能从 UserMessage 边界开始；任务锚点之后可以从任一完整消息块开始；
7. 使用合法起点二分选择满足预算的最长近期尾部，再做协议修复和精确复检；
8. 不创建 ContextSummary，不移动 Session 索引，不改变 Entry Tree 或 working messages。

如果 system prompt、Tool definitions、两个用户锚点和最新消息块组成的最小上下文仍超过
input_budget，则抛出 `ContextWindowExceededError`。Runner 将其映射为：

```text
status      = failed
stop_reason = context_limit
```

Runtime 不 SAVE 工作消息，并保留 PendingInput。Emergency 删除的内容仍完整保留在 Runner
working messages、Checkpoint 和未来 Session SAVE 候选中，只是不进入本次 Provider 请求。

## 10. 失败语义

- 工具链修复和字符裁剪是纯内存确定性操作；内部异常属于 ContextGovernanceError；
- token estimator 的单个实现失败时继续尝试下一个 estimator；
- 所有精确 estimator 失败时允许字符启发式兜底，并在 Report 标明来源；
- 最终仍超过预算时返回 context_limit，不调用 Provider；
- ContextGovernor 失败或超限不修改 working messages、Session 或 PendingInput；
- 不采用 nanobot 的“治理异常后发送几乎未治理原始上下文”兜底，因为它可能把已知超限请求继续
  交给 Provider。

## 11. 不变量

1. 输入 AgentMessage 及其容器不被修改；
2. ContextGovernor 输出只供一次 Provider 请求使用；
3. 本轮初始 UserMessage、最新 UserMessage 和最新完整消息块不被 emergency snip；
4. 每个输出 ToolResult 都有此前声明的同 ID ToolCall；
5. 每个输出 ToolCall 都有同 ID ToolResult，当前尚未执行的调用除外；
6. 不改变工具调用 ID、名称或参数；
7. 不持久化修复占位、压缩占位或裁剪文本；
8. NORMAL 未超过预算时不执行 micro-compaction 或删除普通消息；
9. 最终超过预算时不调用 Provider。
10. 本轮最新 RuntimeStatusMessage 在 Governor 返回后生成，不会被本次治理删除。

## 12. 第一版实施顺序

以下项目已经实现：

- 配置、结果、Report 和异常类型；
- 强类型工具链修复；
- 仅在预算压力下执行的历史工具结果 micro-compaction；
- 首尾字符裁剪；
- PromptTokenEstimator 与 Governor 专用字符兜底；
- 任务锚点、最新用户补充和最近合法尾部组成的 emergency snip；
- 历史与当前 Turn 分别统计的消息删除 Report；
- 无变化 NORMAL 请求的 token 估算复用；
- AgentRunSpec、AgentRunner 和 Runtime 接线；
- 不修改 working messages、协议闭合、预算降级和超限时 Provider 未调用测试。

CLI 通过可选 `--context-window-tokens` 启用 Governor；未提供模型窗口时不猜测模型规格，也不
启用临时治理。`--max-completion-tokens` 和 `--max-tool-result-chars` 用于配置相应预算。

## 13. 暂缓事项

- 文件、Shell、搜索等具体工具的结构化结果视图与确定性摘要；
- 未知 MCP 工具的结构感知结果适配；
- 根据工具元数据自动推导 compactable；
- 模型生成的临时工具结果摘要；
- 多模态 token 估算和内容裁剪；
- Provider context-length 错误后的自适应重试；
- ContextGovernor Report 的 Timeline 持久化；
- checkpoint、pause 和未知工具副作用恢复。

上述工具结果适配属于 Tool Runtime：工具负责理解结果语义，Registry 负责完整 Artifact 和通用
fallback，ContextGovernor 只保留请求级协议修复与预算兜底。当前 Runner 优化不得为了预想中的
新工具协议增加新的领域对象或分支。
