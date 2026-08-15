# 消息插入设计

> 状态：Implemented v0.1  
> 范围：单 Session 运行期间的 PendingInput 注入、批次合并、预算边界与后续 Runner 交接

## 1. 目标

DAO 在 AgentRunner 正在执行时仍允许用户发送后续消息。新消息必须先作为 PendingInput 持久化，
随后要么在安全位置进入当前 Runner，要么保留到当前执行结束后，由它原本触发的提交开启新的 Runner。

设计必须保证：

- 不建立第二套临时消息队列；
- 不丢失、不重复消费用户输入；
- 单个 Runner 不会因持续追问而无限延长；
- 被当前 Turn 消费的每条 PendingInput 都至少进入过一次 Provider 请求；
- Session 正式历史保留每次用户发送的独立身份；
- ContextCheckpoint 与消息注入位置是两个不同概念。

## 2. 名词边界

### 2.1 ContextCheckpoint

ContextCheckpoint 是持久化的执行恢复状态，仍只有：

- `awaiting_tools`；
- `tools_completed`；
- `final_response`。

它解决取消、失败和进程重启后的恢复，不负责决定何时读取新消息。

### 2.2 MessageInjectionPoint

MessageInjectionPoint 是 Runner 中的瞬时检查位置，不落盘：

```python
class MessageInjectionPoint(Enum):
    AFTER_TOOLS = "after_tools"
    AFTER_CANDIDATE_RESPONSE = "after_candidate_response"
```

候选回答只有在第二个注入点没有领取新消息时，才成为真正的 final response。

### 2.3 MessageInjectionBatch

一次注入点读取到的 PendingInput 新增后缀组成一个批次。Runner 只接收由 Runtime 转换好的
UserMessage，不读取 Session，也不依赖 PendingInput：

```python
@dataclass(frozen=True, slots=True)
class MessageInjectionBatch:
    point: MessageInjectionPoint
    messages: tuple[UserMessage, ...]
```

## 3. 唯一消息来源

所有用户输入都先进入 Session 的 PendingInput 队列。运行中追加消息不使用 asyncio.Queue 或 Runner
内部队列。

ExecutionContext 维护已经纳入当前执行的队列前缀：

```python
incorporated_inputs: list[PendingInput]
```

到达注入点时，Runtime 重新读取 Session，并先验证 `id + revision` 前缀完全等于
`incorporated_inputs`，然后选择新增后缀：

```python
new_inputs = session.pending_inputs[len(incorporated_inputs):]
```

一次读取只处理当时的稳定快照。在线性化点之后到达的消息等待下一个注入点。

## 4. 批次合并

PendingInput 和 Session 正式 UserMessage 永不物理合并。每次发送继续保留独立：

- `id`；
- `source_message_id`；
- `revision`；
- 编辑、去重和恢复身份。

同一个尚未进入 Provider 请求的 MessageInjectionBatch 中，连续 UserMessage 只在模型视图中使用
两个换行符合并。批次一旦用于 Provider 请求便关闭；之后到达的输入属于下一个批次。

由于两个真实注入批次之间必然存在 AssistantMessage 或 ToolResultMessage，Provider 视图可以通过
“合并最大连续 UserMessage 段”确定性重建，不修改 Runner 工作消息或 Session 历史。

## 5. Runner 与 Runtime 职责

AgentRunSpec 增加独立的消息注入回调。Checkpoint callback 仍只负责持久化，两者不能合并：

```python
MessageInjectionHandler = Callable[
    [MessageInjectionPoint, int],
    Awaitable[MessageInjectionBatch] | MessageInjectionBatch,
]
```

Runner 负责：

- 只在两个 MessageInjectionPoint 调用回调；
- 计算剩余模型 iteration 和本 Runner 的剩余注入配额；
- 追加返回的强类型 UserMessage；
- 候选回答发生注入时继续下一次 Provider 请求；
- 回调失败时以 `message_injection_error` 结束，不静默吞掉。

Runtime 回调负责：

- 重新读取 Session；
- 校验 active leaf、save cursor 和 PendingInput 前缀；
- 领取不超过 limit 的新增 PendingInput；
- 转换为 UserMessage 并更新 incorporated_inputs；
- 不提前从 Session 删除任何输入。

## 6. 两个注入时序

### 6.1 工具完成后

```text
Provider 产生 ToolCall
→ 保存 awaiting_tools ContextCheckpoint
→ 执行整批工具
→ 追加全部 ToolResultMessage
→ 保存 tools_completed ContextCheckpoint
→ 到达 AFTER_TOOLS
→ 有额度和新增 PendingInput：追加 UserMessage 批次
→ 下一次 Provider iteration
```

### 6.2 候选回答后

```text
Provider 产生候选 AssistantMessage
→ 到达 AFTER_CANDIDATE_RESPONSE
→ 有注入：追加候选 AssistantMessage 和 UserMessage 批次，继续下一 iteration
→ 无注入：保存 final_response ContextCheckpoint，结束 Runner
```

final_response 只描述真正准备提交的终态，避免将仍会继续的候选回答错误标记为终态。

## 7. 双重预算

Runner 同时受两个独立上限约束：

```python
max_turns = 20
max_injected_inputs_per_run = 5
```

`max_turns` 统计 Provider 请求；注入动作本身不增加 iteration。五条配额统计队首启动输入之外，被当前
Runner 吸收的追加 PendingInput 总数，而不是每个注入点的单批上限。

注入必须同时满足：

```text
剩余 Provider iteration > 0
剩余追加消息配额 > 0
```

达到五条配额只停止继续吸收，Runner 仍处理已经纳入的输入，直至模型自然终止或达到 max_turns。

## 8. iteration 耗尽

DAO 不采用 nanobot 在 max_iterations 后 drain 消息、写入历史但不让模型处理的策略。没有剩余
Provider iteration 时：

- 不领取新增 PendingInput；
- 不把它追加进当前工作历史；
- 不消费它；
- 当前 Runner 正常闭合为 limit_reached；
- 剩余输入保留在耐久队列，随后进入新的 Runner。

ContextCheckpoint 的 `next_model_turn` 在恢复时继续原预算，不因注入或重启而重置。

## 9. 新 Runner 交接

普通发送路径是：

```text
submit
→ enqueue_input（先持久化）
→ run_next（等待 Session execution lock）
```

运行中到达的提交先进入 PendingInput，因此当前 Runner 可以看见它；它对应的 `run_next` 同时等待
execution lock：

- 输入被当前 Runner 消费：等待者返回 `injected`，不创建空执行；
- 输入未被吸收：当前 Runner SAVE 并释放锁后，等待者开启新的 Runner。

新 Runner 继承同一 Session 历史并获得新的 max_turns 预算。队首输入始终作为独立的初始问题进入
第一次 Provider 请求，不能和队尾追加消息合并。其余 PendingInput 必须等 Runner 到达第一个真实
MessageInjectionPoint 后，才可组成追加批次；单个 Runner 最多吸收五条。

Runtime 不使用无限 `while pending` 后台轮询。每个用户提交产生有限执行需求，每个 Runner 都独立
结束并释放锁。

## 10. SAVE 不变量

ExecutionContext 区分：

```python
incorporated_inputs: list[PendingInput]
model_seen_input_ids: set[str]
```

Provider 请求发出前，当前请求中的用户消息 ID 进入 model_seen_input_ids。SAVE 只允许消费满足以下
条件的连续队列前缀：

```text
consumed_input_ids == incorporated_input_ids
consumed_input_ids ⊆ model_seen_input_ids
```

`completed` 和 `limit_reached` 可以提交；`failed` 和 `cancelled` 不提交，也不自动重试。

成功 SAVE 后：

1. TurnCommitted 原子消费已处理前缀；
2. SessionStore 持久化；
3. 删除旧 ContextCheckpoint；
4. 剩余 PendingInput 保持不变，等待其已有提交获得 execution lock。

## 11. Checkpoint 多输入身份

ContextCheckpoint 必须记录已经纳入当前执行的 PendingInput `id + revision` 前缀。恢复时该前缀必须
与 Session pending queue 完全一致；任何已纳入输入被编辑都使旧 Checkpoint 失效或冲突。

注入点本身仍不持久化。注入后、下一安全 Checkpoint 前发生崩溃时，消息仍在 PendingInput 中，恢复
后可以重新领取；下一安全 Checkpoint 已包含它时，则依靠多输入身份避免重复注入。

## 12. 对外结果与呈现

当前 Runner 的自然回答即使后面仍有 PendingInput，也是真实有效的阶段回答，应写入 Session 并呈现。
RuntimeResult 需要暴露剩余队列状态：

```python
has_pending_continuation: bool
remaining_pending_count: int
```

- 正常回答且无剩余：作为最终回答呈现；
- 正常回答且有剩余：呈现当前回答，并提示继续处理后续消息；
- max_turns 且无剩余：呈现停止通知；
- max_turns 且有剩余：呈现运行边界通知，随后进入新 Runner。

跨 Runner 不是同一个流式 segment。未来 GUI 可增加瞬时 ContinuationScheduled 事件，不将两个 Runner
的输出拼成同一 Assistant 气泡。

## 13. 验收案例

- 工具完成后注入一批消息，下一 Provider 请求可见；
- 候选回答后注入，候选回答变为中间 AssistantMessage；
- 同批多条 UserMessage 在 Provider 视图合并，Session 中保持独立；
- 多个注入点累计最多吸收五条追加输入；
- 第六条保留 PendingInput，并由新 Runner 处理；
- 最后一次 iteration 不领取新消息；
- SAVE 只消费模型已见输入；
- failed/cancelled 保留所有 PendingInput；
- Checkpoint 恢复不重复注入，任一 incorporated input revision 改变可被识别；
- 被旧 Runner 吸收的并发 submit 返回 injected，而不是 idle；
- 当前结果正确报告 remaining_pending_count。
