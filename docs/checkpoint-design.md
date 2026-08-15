# ContextCheckpoint 设计

> 状态：Implemented v0.1  
> 日期：2026-08-10  
> 范围：单 Session 当前 PendingInput 的安全暂停、崩溃恢复与 SAVE 失败恢复

## 1. 目标

ContextCheckpoint 保存一轮尚未提交对话的技术执行状态，使 DAO 在取消、进程退出或持久化失败后：

- 不丢失已经生成的 AssistantMessage 和 ToolResultMessage；
- 不自动重放完成状态未知的工具；
- 继续处理原来的 PendingInput，不要求用户重新发送问题；
- 最终仍通过一个 TurnCommitted 原子提交完整 Turn；
- 不把恢复状态混入 ContextSummary、Memory 或正式 Entry Tree。

第一版只保存每个 Session 最新的一个 Checkpoint，不实现任意时间点回放、多个并行 Run 或
exactly-once 外部副作用。

## 2. nanobot 参考

nanobot 的 Runner 在三个位置通过 callback 保存状态：

1. `awaiting_tools`：Assistant 工具调用消息已经加入，工具尚未开始；
2. `tools_completed`：整批 ToolResult 已经加入，下一次模型调用尚未开始；
3. `final_response`：最终 Assistant 消息已经形成，Turn 尚未完成外层保存。

Checkpoint 当前存入 Session metadata。取消后，nanobot 会把 AssistantMessage、已完成工具结果和
未完成工具的合成错误写回 Session 历史，然后清除 Checkpoint。这个方案适合其线性消息存储，也
验证了“三个安全点已经足够”的判断。

DAO 保留三个节点和保守的工具恢复原则，但做两项适配：

- Checkpoint 使用独立 Store，不写入 append-only Conversation Event Log；
- 恢复时继续同一个 PendingInput，Checkpoint 尾部保持树外，直到最终 TurnCommitted。

这样不会为了恢复而提前提交半轮历史，也不会让 PendingInput 与 UserMessage 重复。

## 3. 核心类型

```python
CheckpointPhase = Literal[
    "awaiting_tools",
    "tools_completed",
    "final_response",
]

@dataclass(frozen=True, slots=True)
class ContextCheckpoint:
    session_id: str
    input_id: str
    input_revision: int
    base_leaf_id: str | None
    save_cursor: int
    phase: CheckpointPhase
    model: str
    next_model_turn: int
    messages: tuple[AgentMessage, ...]
    tools_used: tuple[str, ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    terminal_status: Literal["completed", "limit_reached"] | None = None
    stop_reason: str | None = None
    final_content: str | None = None
    updated_at: datetime = field(default_factory=utc_now)
```

`messages` 只保存本轮增长的尾部，不复制 Session 历史；其第一条必须是当前 PendingInput 转换出的
UserMessage。`save_cursor` 和 `base_leaf_id` 用于确认恢复时正式历史仍是同一前缀。

不增加 `run_id`。`session_id + input_id + input_revision` 已经标识当前逻辑请求，Checkpoint Store
每次只覆盖该 Session 的最新状态。

Runner 内部使用更轻的 `RunnerCheckpoint`，包含 phase、完整 working messages、模型轮次、usage、
tools_used 和可选终态。Runtime callback 将其裁成 `save_cursor` 后的尾部，并补入 Session/Pending
身份，形成 ContextCheckpoint。

## 4. CheckpointStore

```python
class CheckpointStore(Protocol):
    def load(self, session_id: str) -> ContextCheckpoint | None: ...
    def save(self, checkpoint: ContextCheckpoint) -> None: ...
    def delete(self, session_id: str) -> bool: ...
```

第一版实现：

- `InMemoryCheckpointStore`：单元测试和嵌入式运行；
- `JsonFileCheckpointStore`：CLI 默认持久化实现。

本地实现使用严格 UTF-8、版本化 Codec、session ID 哈希文件名以及 temp + flush + fsync + replace
原子覆盖。Checkpoint Codec 复用 SessionCodec 的 AgentMessage 编解码，因此 ArtifactRef 也能随
ToolResultMessage 正确恢复。

Checkpoint 不放入 JSONL Session Event Log，原因是三个节点会反复替换同一临时状态；如果使用
append-only 事件，每轮都保存不断增长的消息尾部，会产生明显的日志放大。Conversation Log 继续
只保存对话事实，Checkpoint 文件保存可覆盖的运行状态。

CLI 默认目录为 `~/.dao-agent/checkpoints`，并提供 `--checkpoint-dir`。`/clear` 同时删除 Session 和
Checkpoint；Artifact 仍按共享对象规则保留。

## 5. 三个写入节点

### 5.1 awaiting_tools

写入顺序：

```text
Provider 返回完整 ToolCall
→ Runner 追加 AssistantMessage(tool_calls=...)
→ await checkpoint_callback(awaiting_tools)
→ Checkpoint 持久化成功
→ 执行工具批次
```

工具必须在 Checkpoint 保存成功后才能启动。该节点表示：工具可能尚未开始，也可能在持久化之后
执行但没有到达下一个节点，因此崩溃恢复时其完成状态统一视为未知。

### 5.2 tools_completed

```text
所有工具 Task 闭合
→ Runner 追加全部 ToolResultMessage
→ await checkpoint_callback(tools_completed)
→ 下一次 Provider 调用
```

completed、failed、timed_out、reported_error 和 artifact_store 结果都属于“工具调用已经闭合”，
因此可以进入该节点。第一版不在单个并行工具完成时刷新 Checkpoint；若整批中途退出，仍回退到
`awaiting_tools` 的未知状态语义。

### 5.3 final_response

```text
最终 AssistantMessage 已形成
→ await checkpoint_callback(final_response)
→ Runner 返回 AgentRunResult
→ Runtime SAVE
→ Runtime RESPOND
```

普通模型终止、空回答框架兜底和 max_turns 闭合回答都必须保存 final_response。该 Checkpoint 携带
terminal_status、stop_reason 和 final_content，使 Session SAVE 失败后无需再次调用模型。

## 6. 恢复规则

Runtime 在 LOAD 阶段读取当前 Session、队首 PendingInput 和 Checkpoint。

| Checkpoint 状态 | 恢复行为 | 是否再次执行已有工具 |
|---|---|---|
| 无 Checkpoint | 从 Session 历史 + PendingInput 正常开始 | 不适用 |
| awaiting_tools | 为最近 AssistantMessage 的每个未闭合 ToolCall 追加“状态未知”的 ToolResult，然后继续请求模型 | 否 |
| tools_completed | 使用已保存 ToolResult，直接继续下一次模型调用 | 否 |
| final_response | 直接重建终态 AgentRunResult，进入 SAVE | 否 |

`awaiting_tools` 的合成结果使用 `is_error=True`，内容明确说明：执行在工具调用被接受后中断，完成
状态未知，因此 Harness 没有自动重放。它不是对外部副作用成功或失败的判断，模型可以解释风险、
改用其他工具或要求用户确认。

恢复后的 Runner 从 `next_model_turn` 继续，并继承 usage 与 tools_used；最大轮数仍按原始总预算
计算，不因重启而重置。如果已经没有剩余模型轮次，则生成既有 max_turns 闭合回答并进入
final_response。

## 7. 有效性与冲突

Checkpoint 只有同时满足以下条件才可恢复：

- `session_id` 与当前 Session 相同；
- 队首 PendingInput 的 ID 和 revision 完全匹配；
- `base_leaf_id` 与当前 Active Leaf 相同；
- `save_cursor` 等于当前 Active Path 消息数；
- Checkpoint 尾部第一条等于当前 PendingInput.to_user_message()；
- 消息链满足 Assistant ToolCall 与 ToolResult 的关联约束；
- phase 与尾部形态一致。

处理策略：

- PendingInput 已被成功消费：Checkpoint 是提交后的残留，忽略并尽力删除；
- PendingInput 被用户编辑：旧 revision 明确失效，删除后从新内容重新开始；
- Active Leaf 或历史前缀变化：返回 `checkpoint_conflict`，不自动删除或重跑；
- Checkpoint 文件损坏：返回 `checkpoint_corrupt`，不静默从头执行；
- 新消息只追加到 Pending Queue 队尾：不影响当前 Checkpoint。

Runtime checkpoint callback 每次保存前都重新检查 input revision、base leaf 和 save_cursor。运行中
发生编辑或 checkout 时，旧执行不能覆盖新的状态。

## 8. 崩溃一致性

关键顺序与后果：

| 崩溃位置 | 持久事实 | 下次行为 |
|---|---|---|
| 首次 Provider 调用前 | 只有 PendingInput | 重新调用 Provider |
| awaiting_tools 保存失败 | 工具尚未启动 | 返回 checkpoint_error，可安全重试 |
| 工具执行途中 | awaiting_tools | 不重放工具，补未知状态结果 |
| tools_completed 保存失败 | 仍是 awaiting_tools | 所有调用按未知状态处理，不重放 |
| tools_completed 后模型调用途中 | tools_completed | 保留工具结果，重新调用模型 |
| final_response 后、Session SAVE 前 | final_response | 不调用模型，直接重试 SAVE |
| TurnCommitted 成功、Checkpoint 删除前 | Session 已消费 PendingInput | 忽略并清理残留 Checkpoint |

Session commit 与 Checkpoint delete 不需要跨文件事务。提交后的残留可通过 PendingInput 是否仍存在
可靠识别；最危险的方向——Session 未提交却错误删除 Checkpoint——通过“先 commit，后 delete”
避免。

Checkpoint 保存失败不能被忽略：

- awaiting_tools 失败时禁止开始工具；
- tools_completed 失败时禁止继续模型循环；
- final_response 失败时禁止进入 Runtime SAVE/RESPOND。

Runner 将其归一化为 `status="failed"`、`stop_reason="checkpoint_error"`，PendingInput 和上一个已
持久化 Checkpoint 保留。

## 9. Runner 与 Runtime 改动

AgentRunSpec 增加：

```python
checkpoint_callback: CheckpointHandler | None = None
model_turn_offset: int = 0
initial_tools_used: Sequence[str] = ()
initial_usage: Mapping[str, int] = field(default_factory=dict)
```

Runner 仍不读取 Session 或 CheckpointStore，只在三个节点 await callback。

ExecutionContext 增加当前 ContextCheckpoint。PREPARE 负责验证、恢复消息尾部和构造续跑 Spec；RUN
在 final_response 恢复时直接重建 AgentRunResult；SAVE 成功提交完整 Turn 后删除 Checkpoint。

没有配置 CheckpointStore 时，嵌入式 Runtime 保持当前行为；CLI 默认配置持久化 Store。

## 10. 第一版测试矩阵

- 三个节点的 payload 与写入顺序；
- awaiting_tools 保存失败时工具从未执行；
- 工具中途取消后不会自动重放；
- tools_completed 后 Provider 失败可复用已有结果；
- final_response 后 SAVE 失败不重复调用 Provider；
- completed、failed、timeout 和 artifact_store ToolResult 均可恢复；
- ArtifactRef 随 Checkpoint Codec 往返；
- PendingInput 编辑使旧 Checkpoint 失效；
- Active Leaf 改变产生 checkpoint_conflict；
- JSON 文件损坏不会退回从头执行；
- TurnCommitted 成功但 delete 失败时不会重复消费输入；
- 进程重启后 JsonlSessionStore + JsonFileCheckpointStore 联合恢复；
- `/clear` 同时清理 Session 与 Checkpoint。

## 11. 实施顺序

1. ContextCheckpoint、RunnerCheckpoint、phase 和错误类型；
2. CheckpointCodec、InMemoryCheckpointStore、JsonFileCheckpointStore；
3. Runner 三节点 callback 与 turn offset；
4. Runtime LOAD/PREPARE/RUN/SAVE 恢复逻辑；
5. CLI checkpoint-dir 和 `/clear` 联动；
6. 完整恢复与崩溃窗口测试；
7. 更新 M1 验收状态，将 Checkpoint 从规划改为已实现。

消息插入已作为独立的瞬时 MessageInjectionPoint 实现，并与持久化 ContextCheckpoint 保持职责
分离；完整语义见 [消息插入设计](message-injection-design.md)。
