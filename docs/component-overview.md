# DAO Agent Harness 组件设计总览

> 状态：Living Document  
> 最后更新：2026-08-15
> 用途：集中介绍当前组件、职责边界、协作方式与实现状态

## 1. 项目定位

DAO Agent Harness 以 nanobot 的精简 Agent Loop 为行为参考，以 pi agent-core 的 Provider、消息和工具
边界为对照，逐步构建一个小型、可测试、可替换组件的 Agent Harness。

当前策略不是一次性实现完整平台，而是先保持一条可运行的主链路，再把稳定的领域概念
逐步移出 Runner：

```text
外部输入
  → AgentRuntime
  → Session / PendingInput
  → ContextConsolidator / SessionContextResolver
  → ContextBuilder
  → AgentRunner
      ├─ Provider
      └─ ToolRegistry
  → Session SAVE
  → 外部响应
```

其中 Provider、强类型 Runner、ToolRegistry、AgentMessage、Entry Tree Session、JSONL Event Store、
ContextBuilder、ContextSummary、token-budget ContextConsolidator、SkillCatalog、最小 AgentRuntime 和并行
工具批次已经实现；TransportMessage 尚未实现。

内部实现遵守以下精简约束：

- 只有承载独立协议、不变量或可替换边界的概念才建立类型或组件；
- 单次调用且只做参数转发、值搬运的私有包装不保留；
- 阶段函数可以只被调用一次，但必须对应状态机中的明确职责；
- 可选能力通过构造参数注入，不为尚未实现的功能预建 Manager、兼容层或空壳模块；
- 精简不得弱化 Session、Checkpoint、工具调用和消息注入的一致性校验。

顶层 `agent_harness` 包只标识命名空间，不重导出内部类型，也不会在导入时预加载所有组件。
调用方从类型所属模块显式导入，例如：

```python
from agent_harness.runtime import AgentRuntime
from agent_harness.runner import AgentRunner
from agent_harness.tools import ToolRegistry
```

这样公共边界与目录边界一致，也避免一个新组件被无意加入全局兼容承诺。

确定性的 `ScriptedProvider`、`ModelRequest` 和 `FakeTool` 统一位于
`agent_harness.testing`；正式的 `providers` 与 `tools` 子包不公开测试替身。

## 2. 组件状态

| 组件 | 当前状态 | 关键说明 |
|---|---|---|
| Provider contracts | 已实现 | 同时支持非流式和统一流事件 |
| OpenAI-compatible Provider | 已实现 | 支持 JSON completion 与 SSE streaming |
| AgentRunner | 已实现最小版 | 全程使用强类型 AgentMessage，Provider 边界才投影 dict |
| ToolRegistry / AgentTool | 已实现 | 工具标签、参数校验、统一执行结果与工具级 timeout |
| AgentMessage | 已实现 | 已接入 Runner、Runtime 和 Session |
| Session / PendingInput | 已实现 | 正式历史与未完成输入分离 |
| SessionStore | 已实现内存、JSON Snapshot 和 JSONL Event Store | 事件 replay 恢复 Tree、Leaf 和 Pending Queue |
| CLI | 已实现 | 通过 RuntimeRequest、流事件和 JSONL Session 接入 AgentRuntime |
| ContextBuilder | 已实现最小版 | 组装初始提示词和模型视图，不修改工作消息 |
| ContextSummary / Consolidator | 已实现 v1 | 结构化摘要、JSONL 事件、SAVE 后后台探测与 PREPARE 复检 |
| MemoryStore / Dream | 已实现 v0.1 | 强类型 Inbox、严格 MemoryPlan、两阶段 Dream、Workspace MEMORY.md 与后台排空 |
| AgentRuntime | 已实现 | 六阶段状态机、per-Session lock、单输入上限、恢复、注入与暂停修订 |
| 工具批次执行 | 已实现 | Runner 划分连续并行批次和 sequential 屏障 |
| ContextCheckpoint | 已实现 v0.1 | 三节点、独立原子 Store、冲突检测与保守恢复 |
| Message Injection | 已实现 | tools_completed 与 candidate response 双安全点、五条配额与后续 Runner |
| SkillCatalog / Skill tools | 已实现 v0.1 | 渐进加载、资源边界、ToolResult metadata、压缩后恢复 |
| Runtime Status | 已实现 v0.1 | 强类型快照、user-role 投影、Session 持久化与压力治理 |

## 3. Provider

### 3.1 职责

Provider 层隔离模型厂商协议：

- 构造模型请求；
- 把 Agent Harness 工具定义转成厂商格式；
- 解析非流式响应；
- 解析 SSE；
- 在内部聚合被拆分的 tool call ID、名称和 JSON arguments；
- 输出 Provider-neutral 的响应或流事件。

统一非流式结果：

```python
LLMResponse(
    content=...,
    tool_calls=...,
    finish_reason=...,
    usage=...,
)
```

统一流事件：

```text
TextDelta
ToolCallCompleted
ResponseCompleted
```

Runner 不接触 SSE frame、OpenAI chunk、tool call index 或 arguments 分片。只有完整的
`ToolCallCompleted` 才能进入工具执行阶段。

### 3.2 当前实现

- `providers/base.py`：Provider-neutral contracts；
- `providers/openai_compatible.py`：OpenAI-compatible Chat Completions；
- `providers/transport.py`：stdlib JSON HTTP 与 SSE transport；
- `providers/fake.py`：确定性 Runner 测试 Provider。

## 4. AgentRunner

### 4.1 职责

Runner 是内部模型—工具执行循环：

```text
请求模型
  → 得到文本或完整 ToolCall
  → 无工具：完成
  → 有工具：执行并追加 ToolResult
  → 再次请求模型
```

当前 Runner 负责：

- 复制输入消息，维持本次完整工作列表；
- 调用 Provider；
- 聚合流事件为 `LLMResponse`；
- 通过 ToolRegistry 准备工具调用并调度执行；
- 按 `tool_call_id` 生成工具结果；
- 累计 usage；
- 执行最大循环次数限制；
- 返回完整工作消息和终止状态。

Provider 没有 `stream()` 时，流式运行请求自动回退到 `complete()`。

### 4.2 明确不负责

- Session 加载和保存；
- PendingInput 入队和编辑；
- 外部消息接入；
- ContextBuilder 的持久化策略；
- CLI 或 Channel 输出；
- checkpoint、失败重试和外部副作用恢复。

### 4.3 消息边界

`AgentRunSpec.initial_messages` 和 `AgentRunResult.messages` 都使用 AgentMessage。
Runner 每轮调用 Provider 前通过 `ContextBuilder.build_messages()` 创建新的模型视图，
Provider 格式不会进入运行结果或 Session。

## 5. ToolRegistry 与工具

### 5.1 AgentTool

工具最小协议包含：

```python
name
description
parameters
execution_mode  # parallel_safe | sequential
timeout_s       # optional positive seconds
execute(arguments)
```

### 5.2 ToolRegistry

ToolRegistry 是工具说明和实现的唯一来源：

- 按名称注册与查找工具；
- 拒绝重复名称；
- 为 Provider 输出 function schema；
- 根据 schema 安全转换并校验模型参数；
- 保存执行策略需要的 `execution_mode`；
- 可选地通过 ArtifactStore 外置成功的大型文本结果。

统一 Tool Execution Policy 已实现：Registry 通过默认/工具级 timeout、`execute_call()` 和
ToolExecutionResult 负责单调用执行及错误归一化；Runner 只负责批次和并发调度。

ToolOutput 已分离模型视图 `content`、完整 `artifact_content`、显式 `is_error` 与 Harness-only
metadata；完整失败日志也可在保持 `is_error=True` 的同时保存为 Artifact。Registry 对未适配工具
继续提供通用大型结果外置 fallback；SkillCatalog 是
独立只读领域服务；`activate_skill` 与 `read_skill_resource` 作为标准工具注册，具体加载和路径
安全校验委托给 Catalog。Catalog 限制路由 description 的长度；激活结果保留 workspace、user 或
builtin 来源。固定 Tool Contract 保证 Skill 只能提供任务级工作流，不能扩大用户授权或覆盖工具
安全边界。Skill 包的撰写规范见 [DAO Skill Authoring Guide](skill-authoring-guide.md)。

Runner 根据标签把连续的 `parallel_safe` 调用组成并行批次，把 `sequential` 调用作为单独
屏障。普通错误统一形成 `ToolResultMessage(is_error=True)` 并回到模型循环。当前不引入独立
ToolRuntime；统一 timeout 先实现在 Registry 内，未来 approval、sandbox 或远程执行变复杂时，
可以在保持 Runner 接口不变的前提下再提取内部执行器。

大型结果使用内容寻址的 ArtifactRef，不向模型暴露本地路径；`read_artifact` 提供受控分页回读。
ArtifactStore 写入发生在 Session SAVE 之前，保证已提交引用可读。详细规则见
[ArtifactStore 设计](artifact-store-design.md)。

基础文件工具开始采用 pi coding-agent 的成熟行为协议。共享 `ToolPathPolicy`、UTF-8 字节/完整行
截断和文件修改队列已经实现；`read/ls/find/grep/write/edit/bash` 七个 pi 风格基础工具均已加入并由
CLI 默认注册。当前采用直接 Python 实现，优先展示 Harness 的完整工具边界，不提前实现完整
`.gitignore`、模糊编辑、Shell 进程树、流式输出、approval 或 sandbox。详细协议见
[基础工具设计](basic-tools-design.md)。

## 6. AgentMessage

### 6.1 类型

```python
AgentMessage = UserMessage | AssistantMessage | ToolResultMessage | RuntimeStatusMessage
```

辅助结构：

```python
ToolCall(id, name, arguments)
```

第一版只支持文本 content。消息 dataclass 不可变，Session 通过复制列表建立运行副本，不需
深拷贝每条消息。

### 6.2 身份规则

- 每条消息有稳定 `message.id`；
- `PendingInput.id` 同时成为成功提交后的 `UserMessage.id`；
- ToolResult 必须携带 `tool_call_id`；
- 消息不保存 `run_id`、UserTurn 或 ModelStep ID；
- execution 身份只属于临时 ExecutionContext 或未来运行日志；
- ProviderMessage 不进入 Session。

## 7. Session 与 PendingInput

### 7.1 数据边界

```python
Session(
    id=...,
    entries=...,         # 包含 parent_id 的 Message Entry Tree
    active_leaf_id=...,  # 当前上下文分支末端
    pending_inputs=...,  # 已接受但尚未加入正式消息树的用户输入
    metadata=...,
)
```

```python
PendingInput(
    id=...,
    source_message_id=...,
    content=...,
    created_at=...,
    edited_at=...,
    revision=...,
)
```

PendingInput 只保存用户意图和编辑 revision，不保存 `attempts`、`last_error` 或其他 execution
状态。队列中相同 `source_message_id` 的重复输入会复用已有 PendingInput。

“先保存”表示 Runner 启动前已经追加并同步 `InputEnqueued`。`JsonlSessionStore` 可以在进程
重启后 replay Pending Queue、Message Entry Tree 和 Active Leaf。

### 7.2 运行副本与 save_cursor

```python
working_messages = session.copy_history()
base_leaf_id = session.active_leaf_id
save_cursor = len(working_messages)
working_messages.append(pending_input.to_user_message())
```

Runner 只修改 working messages。执行 `completed` 或 `limit_reached` 后：

```python
session.commit_working_messages(
    working_messages=result.messages,
    base_leaf_id=base_leaf_id,
    save_cursor=save_cursor,
    consumed_input_ids=(pending_input.id,),
)
session_store.save(session)
```

commit 产生单个 `TurnCommitted`，其中同时包含完整有序消息段、消费的输入 revision 和新的
active leaf。`failed` 和 `cancelled` 不调用 commit，正式历史和 PendingInput 保持不变。`limit_reached`
由 Runner 追加框架终止消息后提交，避免历史以未闭合的工具批次结束。

### 7.3 已实现的不变量

- 消费的 input ID 必须是 pending 队列的连续前缀；
- working messages 的已有历史前缀不能改变；
- active leaf 不能在 execution 期间变化；
- 每个消费的 PendingInput 必须在保存尾部恰好出现一次；
- 保存后的 Entry ID 和消息 ID 不得在任何分支重复；
- 后来入队、但未消费的 PendingInput 必须保留；
- 编辑 PendingInput 不修改正式历史；
- PendingInput 被编辑后，携带旧内容的迟到 execution 不能提交。

## 8. SessionStore

最小 Protocol：

```python
get_or_create(session_id) -> Session
save(session) -> None
delete(session_id) -> bool
```

`InMemorySessionStore` 服务单元测试；`JsonFileSessionStore` 保留版本化原子 Snapshot；
`JsonlSessionStore` 是当前追加式耐久实现。后者保存 `InputEnqueued`、`InputEdited`、
`TurnCommitted` 和 `LeafChanged`，并通过 replay 物化 Session。`TurnCommitted` 使用一个
JSONL record 同时提交消息段、消费 PendingInput 和移动 Leaf。损坏的中间记录会报错，崩溃
留下的无换行尾记录可以安全截断。数据库事务、多进程锁与 lease 暂缓。详见
[Session Entry Tree 与事件持久化设计](session-persistence-design.md)。

## 9. CLI

当前 CLI：

- 创建 OpenAI-compatible Provider；
- 注册 UTC 时间工具和受控分页读取的 `read_artifact`；
- 使用 LocalArtifactStore 外置大型工具结果；
- 通过 RuntimeRequest 调用 AgentRuntime，不维护本地 history；
- 使用 JsonlSessionStore 持久化多轮对话；
- 消费分段 RuntimeStreamEvent 实时输出；
- 支持运行中追加以及 `/pause`、`/edit`、`/resume`、`/retry`、`/clear` 和 `/exit`。

DAO v1 只实现 CLI Adapter。CLI 不再直接持有 history，而是提交
RuntimeRequest、消费临时 RuntimeStreamEvent，并以 SAVE 后返回的 RuntimeResult 确认终态。
协议详见 [CLI 与 Runtime 对外协议](cli-runtime-protocol.md)。

## 10. ContextBuilder 与后续组件

### 10.1 ContextBuilder

ContextBuilder 根据完整 `working_messages` 生成初始 `ModelContext`：

- 按固定顺序组装 Identity、Bootstrap Files 和 Tool Contract；
- 将强类型 AgentMessage 转换为模型消息；
- 接受 Memory、MCP 和其他组件提供的额外系统片段；
- 不修改 working messages；
- 不改变 save_cursor；
- 不保存 Session。

ContextBuilder 已接入 SkillCatalog 目录，并能从摘要覆盖的历史派生受保护 Skill 激活前缀。
Runtime Status 由 Runner 构造，ContextBuilder 只把 RuntimeStatusMessage 投影为标准 user role；
它自身仍不负责多模态、token budget、修复、裁剪或压缩。
工具循环中的逐轮治理由 Runner ContextGovernor 负责，现已实现强类型工具链修复、显式
micro-compaction、单结果首尾裁剪和受当前 Turn 保护的 emergency snip，详见
[Runner ContextGovernor 设计](context-governor-design.md)。

压缩实现不会把 Summary 放入 Conversation Entry Tree。当前由树外
`ContextSummary` 保存分支相关结构化摘要和 `covered_through_entry_id`，
`SessionContextResolver` 解析为 `summary + 未覆盖消息尾部`，ContextBuilder 再把 Summary 作为
`Archived Conversation Context` 系统模块注入。Summary 通过非树 SessionEvent 耐久化，不
改变 Active Leaf；详细规则见 [Session Entry Tree 与事件持久化设计](session-persistence-design.md)。
v1 压缩算法直接采用 nanobot Consolidator 的预算公式、50% 目标比例、User Turn 安全边界和
最多 5 轮循环，只做 Entry Tree 与失败语义所需的适配。Summary 内容使用固定强类型结构；在
Provider 尚无原生 JSON Schema 能力时，通过 `LLMResponse.content` 返回 JSON 文本，并在 Harness
中执行严格解析和一次修复。

长期记忆采用 nanobot 的分层思路：ContextConsolidator 每次成功覆盖新的历史消息块后，将该增量
写入耐久 Memory Inbox；后台 Dream 先生成严格结构化 MemoryPlan，再通过只允许访问
`memory/MEMORY.md` 的隔离 Runner 做局部修改。Memory 属于 Workspace，不属于 Session Entry Tree；
ContextSummary 服务当前分支连续性，Memory 服务跨 Session 复用，ContextCheckpoint 服务执行恢复。
详细协议见 [长期记忆与 Dream 设计](memory-system-design.md)。

### 10.2 AgentRuntime

AgentRuntime 是外层编排器：

```text
接收输入
→ PendingInput 入队并保存
→ 获取单 Session execution 权限
→ 复制历史并记录 save_cursor
→ 构建上下文
→ 调用 Runner
→ completed 或 limit_reached 时保存尾部并确认 PendingInput
→ 返回外部响应
```

同一个 Session 同时只运行一个 execution，但活跃执行期间仍允许新输入进入 pending 队尾。
Runtime 还维护进程内 ActiveRun 和 paused latch。用户主动暂停时根据目标 PendingInput ID 截断
未提交工作消息；初始问题修订回到本轮起点，追加消息修订只回退该消息及其派生结果。

### 10.3 工具执行边界

工具自身声明 `parallel_safe` 或 `sequential`，并可声明 timeout；ToolRegistry 负责注册、schema、
参数准备、单次执行、deadline 与统一结果；Runner 负责批次划分、并发调度和 ToolResultMessage
转换。结果按 `tool_call_id` 关联，并行结果允许按完成顺序进入工作消息。独立 ToolRuntime 不属于
当前架构的必需组件。Execution Policy v0.2 已实现。

### 10.4 应用 Adapter

第一版不定义通用 InboundMessage / OutboundMessage，只实现更薄的 CLI Adapter。GUI、HTTP、
WebSocket 或聊天渠道在后续版本通过独立 Adapter 接入；Runner 始终不接触应用层消息协议。

## 11. 暂缓的横向能力

- 运行事件存储和 tracing；
- 模型层以外的失败重试策略；
- 外部工具副作用 exactly-once；
- RAG 与基于相关性的 Memory 检索；
- 多进程 Session 协调；
- 多模态 AgentMessage；
- 非 Message Entry、旧消息内容编辑和分支合并。
- 基于当前上下文、但不提交主 Session 的临时提问窗口。

消息插入已经在整批工具完成后和候选最终回答完成后实现。它复用耐久 PendingInput，限制每个
Runner 最多五条追加输入，并且不会在模型流式生成途中插入。

主动暂停与局部修订已经实现。暂停工作前缀暂不持久化；PendingInput 编辑仍会作为 SessionEvent
保存。被截断工具调用的外部副作用只做 `completed/uncertain` 风险报告，暂不自动补偿。

## 12. 当前实现目录

```text
src/agent_harness/
├─ context.py
├─ injection.py
├─ messages.py
├─ skills.py
├─ session.py
├─ status.py
├─ status_builder.py
├─ runner.py
├─ runtime.py
├─ cli.py
├─ memory/
├─ providers/
├─ tools/
├─ storage/
└─ templates/
```

详细设计与实施路线：

- [Runner 设计](runner-design.md)
- [工具注册、校验与执行设计](tool-execution-design.md)
- [基础工具设计](basic-tools-design.md)
- [ArtifactStore 设计](artifact-store-design.md)
- [Skill 系统设计](skill-system-design.md)
- [ContextBuilder 设计](context-builder-design.md)
- [Runtime Status 设计](runtime-status-design.md)
- [Agent Runtime 与 Session 设计](runtime-design.md)
- [消息插入设计](message-injection-design.md)
- [主动暂停与消息修订设计](pause-revision-design.md)
- [Session 持久化设计](session-persistence-design.md)
- [长期记忆与 Dream 设计](memory-system-design.md)
- [提取与实施清单](extraction-checklist.md)

## 13. 当前验证

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

测试覆盖 Provider 翻译与流式聚合、typed Runner 正常与错误路径、工具参数校验、并行批次与
串行屏障、消息不变量、Session 入队/编辑/TurnCommitted、Runtime 保存判定、per-Session
并发行为、Entry Tree 分支、强类型事件 replay、Snapshot 迁移、JSONL 尾部恢复，以及
ContextSummary 的严格解析、修复、分支解析、持久化、Runtime 上下文投影，以及
ContextGovernor 的工具链修复、临时裁剪、ArtifactStore、Registry 大结果外置、Session Codec v6
和 Runtime Status 的生成、治理、持久化及摘要过滤。
ContextCheckpoint 三节点、独立 Store、SAVE 失败恢复，以及消息插入双安全点、五条配额、多输入
Checkpoint、后续 Runner 交接、主动暂停、局部消息修订、单输入上限、后台压缩协调和 CLI 重启
以及 SkillCatalog、资源路径安全、Skill 工具、metadata 持久化、压缩后恢复、Memory Inbox、Dream
失败游标、长期记忆注入和后台排空也已覆盖。当前共 276 个测试。
