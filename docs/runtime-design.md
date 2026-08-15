# Agent Runtime 设计

> 状态：Implemented v0.9  
> 最后更新：2026-08-11  
> 范围：最小 Harness 的输入入队、Session 编排、Runner 调用、保存与响应

## 1. 组件边界

系统分为外层 `AgentRuntime` 和内层 `AgentRunner`：

- Runtime 管理 PendingInput、Session、执行互斥、阶段推进和 SAVE；
- ContextBuilder 组装系统提示词，并把强类型消息投影为 Provider 消息；
- Runner 只执行一次模型—工具循环并返回完整工作消息；
- Provider 负责厂商协议、流式分片和工具调用参数的归一化；
- SessionStore 负责 Session 的同步读写。

```text
enqueue_input
    ↓
LOAD → PREPARE → RUN → SAVE → RESPOND → DONE
                       ↑
                 AgentRunner
```

Runner 不读取 SessionStore；Runtime 不加工厂商响应，也不执行工具。

## 2. 状态机

第一版只有六个阶段：

```python
class ExecutionPhase(Enum):
    LOAD = auto()
    PREPARE = auto()
    RUN = auto()
    SAVE = auto()
    RESPOND = auto()
    DONE = auto()
```

转换表固定为：

| 当前阶段 | 阶段结果 | 下一阶段 |
|---|---|---|
| LOAD | `ok` | PREPARE |
| LOAD | `no_pending` | RESPOND |
| PREPARE | `ok` | RUN |
| RUN | `ok` | SAVE |
| SAVE | `ok` | RESPOND |
| RESPOND | `ok` | DONE |

`AgentRunResult.status` 是 RUN 产生的数据，不是状态机转换事件。阶段抛出的异常直接向调用者
传播；尤其 SAVE 失败后不能继续 RESPOND，避免向外确认尚未持久化的回答。

## 3. ExecutionContext

ExecutionContext 只保存跨阶段必需的数据：

```python
@dataclass(slots=True)
class ExecutionContext:
    session_id: str
    phase: ExecutionPhase = ExecutionPhase.LOAD
    pending_input: PendingInput | None = None
    base_leaf_id: str | None = None
    save_cursor: int = 0
    run_spec: AgentRunSpec | None = None
    run_result: AgentRunResult | None = None
    runtime_result: RuntimeResult | None = None
    incorporated_inputs: list[PendingInput] = field(default_factory=list)
    model_seen_input_ids: set[str] = field(default_factory=set)
    paused_run: _PausedRunState | None = None
```

不加入 `run_id`、trace、outbound、callback 或 `should_save`。这些都不是最小执行所需的领域
状态。ExecutionContext 临时存在，不进入 Session。

### 3.1 Runtime 内部编排收口

Runtime 仍保留在一个模块中，也不新增领域对象；长流程通过行为保持型私有函数拆分。公开入口和
阶段函数只展示控制流：

```text
run_next
→ short-circuit paused / injected
→ register ActiveRun
→ drive state machine
→ materialize pause or return RuntimeResult

PREPARE
→ validate Session
→ restore incorporated inputs
→ enforce per-input token limit
→ await post-SAVE consolidation（默认预留 2048 input tokens）
→ recheck with the real PendingInput（不重复扣除预留）
→ build working messages
→ build AgentRunSpec
```

具体的 Checkpoint 持久化、消息注入领取、流事件分段、暂停消息截断和副作用分析分别由独立私有
函数完成。它们没有自己的生命周期和持久化身份，因此不提取新的 Coordinator 类。这样既保持
六阶段主逻辑可读，也避免为单一调用链增加无必要抽象。

## 4. 各阶段职责

### 4.1 LOAD

同步读取或创建 Session。如果 pending 队列为空，返回 `no_pending`；否则记录队首
PendingInput 的不可变快照、当前 `active_leaf_id`，以及当前分支消息数作为 `save_cursor`。

### 4.2 PREPARE

再次同步读取 Session，确认 active leaf、当前分支长度和队首 PendingInput 与 LOAD 取得的
快照一致。可选 ContextConsolidator 先等待同 Session 的 SAVE 后后台探测结束，再携带真实
PendingInput 复检一次。在此之前，可选单输入 Token Gate 会检查尚未被模型处理的 PendingInput；
超限时返回 `failed/input_too_large`，保留队列供编辑，不等待压缩也不调用 Provider。随后重新取得
Session 并再次验证游标。最后：

1. `working_messages = session.copy_history()`；
2. 追加 `pending_input.to_user_message()`；
3. 只构建一次 system prompt；
4. 生成使用强类型 `AgentMessage` 的 `AgentRunSpec`。

Runtime 在 PREPARE 与 RUN 之间不持有 Session 引用。运行期新输入可以继续追加到 pending
队尾，不会改变本次工作消息。

### 4.3 RUN

调用 `AgentRunner.run(run_spec)`，把结果保存到 ExecutionContext。无论 Runner 返回
`completed`、`limit_reached`、`failed` 或 `cancelled`，阶段本身都返回 `ok`，交给 SAVE
统一判断。

### 4.4 SAVE

SAVE 重新取得最新 Session，从而保留 RUN 期间后来入队的输入。

| Runner 状态 | Session 行为 |
|---|---|
| `completed` | 用一个 TurnCommitted 提交完整消息尾部并消费本次 PendingInput |
| `limit_reached` | 同上；Runner 必须已经追加框架终止消息，保证尾部闭合 |
| `failed` | 不修改 Session，PendingInput 保留 |
| `cancelled` | 不修改 Session，PendingInput 保留 |

提交统一调用 `Session.commit_working_messages()`，由 Session 校验：

- base_leaf_id、save_cursor 和当前分支历史前缀未发生变化；
- 消费 ID 是 pending 队列连续前缀；
- PendingInput 内容和 revision 没有在执行期间被替换；
- 本轮新增 Entry ID 和消息 ID 在所有分支中唯一。

校验通过后 Session 产生单个 `TurnCommitted` 事件；`SessionStore.save()` 负责追加并 fsync。
成功持久化后 Runtime 立即调度一次受跟踪的后台 Consolidator 探测，不等待摘要生成便进入
RESPOND。后台任务按 Session 去重；下一轮 PREPARE 会等待同一任务，CLI 清空 Session 前取消它，
退出时排空剩余任务。

### 4.5 RESPOND

RESPOND 只把内部结果映射为 `RuntimeResult`：

- 没有 pending 输入时返回 `idle`；
- 其他情况保留 Runner 的 status、final_content、stop_reason、tools_used、usage 和 error；
- 不发送 Channel 消息，不修改 Session。

### 4.6 DONE

返回已构造的 RuntimeResult，不再产生副作用。

## 5. 公开接口

公开接口已经使用强类型 RuntimeRequest 和每次调用的流事件处理器。完整协议见
[CLI 与 Runtime 对外协议](cli-runtime-protocol.md)。

```python
class AgentRuntime:
    def enqueue_input(
        self,
        session_id: str,
        source_message_id: str,
        content: str,
    ) -> PendingInput: ...

    async def run_next(self, session_id: str) -> RuntimeResult: ...

    async def submit(
        self, request: RuntimeRequest,
    ) -> RuntimeResult: ...

    async def pause_for_revision(
        self, session_id: str, input_id: str | None = None,
    ) -> RuntimeResult: ...

    def revise_paused_input(
        self, session_id: str, input_id: str, content: str,
    ) -> PendingInput: ...

    async def restart_pending(self, session_id: str) -> RuntimeResult: ...
```

`enqueue_input()` 在启动 Runner 之前把用户输入保存到 pending 队列。同一 Session 内相同
`source_message_id` 的重复提交复用已有 PendingInput。

## 6. RuntimeResult

```python
RuntimeStatus = Literal[
    "idle",
    "completed",
    "failed",
    "cancelled",
    "limit_reached",
    "injected",
    "paused",
]

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
    has_pending_continuation: bool = False
    remaining_pending_count: int = 0
    revision_target_input_id: str | None = None
    discarded_message_count: int = 0
    side_effect_status: Literal["none", "completed", "uncertain"] = "none"
    discarded_tool_call_ids: tuple[str, ...] = ()
```

PendingInput.id 已足够标识一次用户逻辑请求，因此不额外暴露或持久化 run_id。

## 7. 并发与锁

第一版每个 Session 只有一个 `asyncio.Lock`，称为 execution lock：

- 同一 Session 的 `run_next()` 串行执行；
- 不同 Session 可以并行；
- `enqueue_input()` 仍可在 RUN await 期间同步追加队尾；
- SessionStore 当前是同步接口，所有 load—modify—save 代码块中不能出现 await。

不增加长期持有的 mutation lock。主动暂停使用 paused latch 阻止 execution lock 的等待者启动，
PendingInput 编辑仍是同步 load—modify—save 临界区。未来引入异步数据库 Store 或多进程写入时，
再把 Session 变更升级为事务或独立短锁。

## 8. 消息和 ContextBuilder 边界

Runtime、Runner 和 Session 全程使用 `AgentMessage`：

```text
Session.active_messages()（完整 SAVE 历史）
  → SessionContextResolver（Summary + 未覆盖尾部）
  → working AgentMessage[] + model_message_start
  → AgentRunner
  → ContextBuilder.build_messages()
  → Provider Mapping[]
```

system prompt 在 PREPARE 中构建一次。可选 ContextConsolidator 通常已在上一轮 SAVE 后后台按 token
预算生成耐久 Summary；PREPARE 使用真实输入复检，
Runner 每次调用 Provider 前从稳定的 `model_message_start` 重新投影当前工作消息，
所以工具结果自然进入下一次模型请求，而 Provider 格式永远不会写回 Session。

## 9. 失败与一致性

- Provider 抛出异常或返回 `finish_reason="error"`：Runner 返回 `failed`，不追加虚假回答；
- 普通工具异常：转换为 `ToolResultMessage(is_error=True)`，继续循环；
- 空的正常模型回答：Runner 写入稳定的框架兜底文本；
- 达到 max_turns：Runner追加稳定的框架终止文本并返回 `limit_reached`；
- SAVE 冲突：异常向外传播，不返回成功结果；
- 失败和取消都保留 PendingInput，用户无需重新发送。
- 单条输入超过 `max_input_tokens`：返回 `input_too_large`，保留 PendingInput 和修订目标；
  TokenEstimator 不可用时使用 UTF-8 字节数作为保守上界；

## 10. 已完成的恢复能力与暂缓事项

- ContextCheckpoint 三个安全节点、独立 Store 与崩溃恢复已实现，见
  [ContextCheckpoint 设计](checkpoint-design.md)；
- PendingInput 安全点消息插入、五条总配额与后续 Runner 交接已实现，见
  [消息插入设计](message-injection-design.md)；
- Runtime 事件、trace 和 diagnostics；
- 主动暂停、PendingInput 局部修订与进程内续跑已实现，见
  [主动暂停与消息修订设计](pause-revision-design.md)；
- 长期记忆、MCP，以及 Skill Package、安装和 Authoring；
- 工具 approval 和 sandbox 执行策略；
- 异步或数据库 SessionStore、多进程 lease；
- GUI、HTTP、Channel Adapter 和通用 TransportMessage；CLI 已按 RuntimeRequest /
  RuntimeStreamEvent 协议接线。

消息插入固定使用两个瞬时检查位置：整批工具完成后，以及候选最终回答生成后。它们不属于
ContextCheckpoint，也不在模型流式生成途中读取队列。

## 11. 当前实施状态

以下 Runtime 主链路已经实现并有测试覆盖：

- ExecutionPhase、ExecutionContext 和显式转换表；
- PendingInput 事件先保存、active branch 复制、base_leaf_id 和 save_cursor；
- PREPARE 构建 system prompt 与 typed AgentRunSpec；
- RUN、SAVE 判定和 RuntimeResult 映射；
- 同 Session 串行、不同 Session 并行；
- RUN 期间新输入入队和 SAVE 时保留队尾；
- Provider 失败、取消与 limit_reached 的 Session 行为。
- ContextSummary 在 SAVE 后后台预检并生成，PREPARE 等待后使用真实 PendingInput 复检；
  完整历史仍可通过 SAVE 校验。
- 初始输入和运行中追加输入共用单条 Token 上限；超限追加消息不注入当前 Runner，并保持队列顺序。
- 三节点 ContextCheckpoint 写入、校验、恢复与成功提交后清理；
- 状态未知工具的保守恢复，以及 final_response 后 SAVE 失败的无模型重放重试。
- 多 PendingInput 前缀领取、模型已见校验、同批模型视图合并和并发 submit 的 injected 结果；
- iteration 或五条追加配额耗尽后保留未处理 PendingInput，由后续有界 Runner 处理。
- ActiveRun、paused latch、按 UserMessage.id 截断、修订后续跑和工具副作用风险报告。
- `run_next`、PREPARE 与 pause materialization 只按行为边界拆分；单次转发包装已移除，
  外部协议不变。

CLI 已通过 RuntimeRequest、RuntimeStreamEvent 和 RuntimeResult 接入 AgentRuntime，并使用
JsonlSessionStore 延续多轮对话。JSONL SessionEvent replay、Message Entry Tree 和 Leaf 切换
已经实现；CLI 也已支持运行中追加、`/pause`、`/edit` 与 `/resume`。数据库 Store、通用
TransportMessage 和跨进程暂停快照仍不属于当前实现。
