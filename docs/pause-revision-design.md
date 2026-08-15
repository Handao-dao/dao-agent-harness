# 主动暂停与消息修订设计

> 状态：Implemented v0.1  
> 最后更新：2026-08-11  
> 范围：单进程 Runtime 内主动暂停、PendingInput 修订、局部工作历史回退与 CLI 交互

## 1. 目标与语义

DAO 区分两类中断：

- 系统失败、进程崩溃或非用户取消：保留 ContextCheckpoint，按恢复协议继续；
- 用户主动暂停：不恢复目标输入及其派生结果，而是让用户修改 PendingInput 后重新运行。

如果当前 Turn 没有追加消息，默认修订最初的 PendingInput；如果已经追加消息，默认修订最新一条
追加 PendingInput。追加意味着用户认可此前问题和模型进度，只希望补充信息，因此不应默认清空
整个 Turn。调用方也可以显式传入 `input_id`，选择重写更早的输入。

## 2. 消息本身就是回退边界

PendingInput 转换为 UserMessage 时沿用同一个 ID。运行消息仍保存独立 UserMessage，只在 Provider
视图中合并连续用户消息。因此不引入 InjectionRevisionBoundary 或新的消息类型：

```text
target_index = messages.index(UserMessage(id=input_id))
preserved = messages[:target_index]
discarded = messages[target_index:]
```

目标消息尚未进入运行消息时，`discarded` 为空，已完成工作前缀全部保留；目标消息已经被模型看到
时，删除目标及其后的 Assistant、ToolCall、ToolResult 和其他追加输入。修改后的 PendingInput 以新
revision 重新生成 UserMessage。

## 3. 最小运行控制

Runtime 每个 Session 维护三项内存状态：

- `_active_runs`：当前 asyncio Task 与 ExecutionContext；
- `_paused_sessions`：阻止等待中的 submit/run_next 启动新 Runner 的暂停锁存；
- `_paused_runs`：取消后截断得到的工作消息前缀和最小续跑数据。

`_paused_runs` 不是故障恢复 Checkpoint，也不进入 Session。它只服务当前进程内的交互修订。

暂停顺序固定为：

```text
选择 revision target
→ 设置 paused latch
→ 取消并等待 ActiveRun
→ 确认目标仍是 PendingInput
→ 按 UserMessage.id 截断
→ 删除旧 ContextCheckpoint
→ 进入 paused 状态
```

paused latch 必须先于取消设置。否则同一 Session 中等待 execution lock 的 submit 可能在活动任务
退出后立即启动新的 Runner。

如果 SAVE 已经完成，目标输入已不在 PendingInput 队列，暂停返回
`stop_reason="completed_before_pause"`，不得修改已提交 Entry Tree。修改历史消息应通过未来的
Entry Tree 分支能力完成。

## 4. 公开接口

```python
async def pause_for_revision(
    session_id: str,
    input_id: str | None = None,
) -> RuntimeResult: ...

def revise_paused_input(
    session_id: str,
    input_id: str,
    content: str,
) -> PendingInput: ...

async def restart_pending(
    session_id: str,
    *,
    on_stream: RuntimeStreamHandler | None = None,
) -> RuntimeResult: ...

async def revise_and_restart(...) -> RuntimeResult: ...
```

暂停后只允许编辑选中的 revision target。新输入仍可进入 PendingInput 队尾，但 `run_next()` 在
paused latch 清除前只返回 `status="paused"`，不会启动 Provider。

RuntimeResult 为暂停增加：

```python
revision_target_input_id: str | None
discarded_message_count: int
side_effect_status: Literal["none", "completed", "uncertain"]
discarded_tool_call_ids: tuple[str, ...]
```

## 5. 续跑状态

截断后从消息前缀推导续跑数据：

- 保留的 PendingInput ID 是截断前 UserMessage 在 pending 队列中的有序前缀；
- `model_turn_offset` 等于当前 Turn 保留的 AssistantMessage 数；
- `tools_used` 从保留 AssistantMessage 的 ToolCall 重新推导；
- 被截断的模型迭代不再占用 max_turns；
- token usage 是已经真实发生的调用成本，不因用户回退而退还，续跑继续累计。

PREPARE 验证 Session active leaf、save_cursor 和 PendingInput 前缀仍匹配，然后把修改后的目标输入
重新加入工作消息。连续用户消息仍由 ContextBuilder 在下一次 Provider 请求时生成合并视图。

## 6. 工具副作用

删除 ToolCall/ToolResult 只能回退 Harness 内部消息，不能撤销外部操作。第一版做保守报告：

- 截断后没有 ToolCall：`none`；
- ToolCall 有对应 ToolResult：`completed`，表示工具执行已完成但副作用可能保留；
- ToolCall 没有对应 ToolResult：`uncertain`，表示取消发生在工具执行期间。

第一版不做补偿事务。未来由工具声明 `read_only`、`idempotent`、`mutating` 或
`compensatable` 等能力，再增加确认、幂等键、撤销函数和副作用日志。

## 7. 持久化边界

PendingInput 的创建和编辑继续通过 SessionEvent 持久化，用户内容不会丢失。暂停工作前缀只在
内存中保留；如果进程在 paused 状态退出，下次启动将从上一轮已提交 Session 和当前 PendingInput
重新运行，尚未提交的模型进度会丢失。

这个约束保持 v0.1 简单，也避免把交互修订状态混入 ContextCheckpoint。未来确有跨进程暂停需求
时，再设计独立的耐久 WorkingTurn Snapshot。

## 8. CLI 行为

CLI 在模型运行期间继续读取 stdin：

- 普通文本进入 PendingInput 队尾，等待安全点注入；
- `/pause` 默认选择最新追加输入并暂停；
- 暂停后直接输入文本或使用 `/edit <text>`，都会修改目标并重新运行；
- `/resume` 不修改内容，直接从保留前缀继续；
- `/exit` 等待当前执行结束；需要中止时先使用 `/pause`；
- `/clear` 会先暂停活动执行，再删除 Session、Checkpoint 和内存暂停状态。

## 9. 已实现测试

- 初始输入暂停、修改和从已提交历史重跑；
- 已被模型看到的追加输入局部截断，保留此前模型回答；
- 尚未被模型看到的最新追加输入默认成为修订目标；
- paused latch 阻止等待中的同 Session submit 自动启动；
- 只允许修改选中的 PendingInput；
- 正在执行的工具形成 `uncertain` 副作用提示；
- CLI `/pause`、直接替换文本和自动重启端到端路径。
