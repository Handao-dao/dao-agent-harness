# Session Entry Tree 与事件持久化设计

> 状态：Implemented v0.3  
> 最后更新：2026-08-10

## 1. 当前决定

Session 不再以平铺 `messages` 列表作为事实来源，而是由追加式 `SessionEvent` 日志重放得到：

```text
JSONL Session Event Log
        ↓ replay
Session Projection
├─ pending_inputs
├─ Message Entry Tree
└─ active_leaf_id
```

强类型领域对象仍然是 Runtime 和 Runner 的接口；JSONL 只保存带稳定判别字段的 JSON 数据。

## 2. Session 投影

```python
Session(
    entries=list[MessageEntry],
    active_leaf_id=str | None,
    pending_inputs=list[PendingInput],
    metadata=dict,
)
```

```python
MessageEntry(
    id=...,                 # 树节点身份
    parent_id=...,          # 父节点；根节点为 None
    message=AgentMessage,   # 消息自身仍有独立 message.id
)
```

`Session.messages` 是当前 active leaf 到根节点路径的物化视图，不再是可直接修改的存储列表。
`checkout(entry_id)` 只移动 active leaf，不删除任何分支。

## 3. PendingInput 的语义

PendingInput 是已经被系统接受、但尚未确定或提交到正式对话树位置的用户输入。它满足两个目标：

- Runner 失败时用户输入仍可恢复，不需要重新发送；
- Runner 活跃期间可以继续追加消息到队尾。

运行期间追加的输入不能立刻成为 MessageEntry，因为当前回答最终可能继续追加 Assistant、
ToolResult 等节点。新输入的正确 `parent_id` 只有轮到它执行时才能确定。

PendingInput 增加单调递增的 `revision`。用户编辑会产生新 revision；旧 execution 携带的消息
内容无法通过 SAVE 校验，因而不能提交迟到结果。

## 4. SessionEvent

当前持久化五类领域事件：

```python
SessionEvent = (
    InputEnqueued
    | InputEdited
    | TurnCommitted
    | LeafChanged
    | ContextSummaryCreated
)
```

### 4.1 InputEnqueued

用户输入被接受时立即追加并 `fsync`：

```json
{
  "type": "input_enqueued",
  "id": "event-1",
  "timestamp": "...",
  "input": {
    "id": "input-1",
    "source_message_id": "external-1",
    "content": "用户问题",
    "revision": 1
  }
}
```

### 4.2 InputEdited

只追加编辑操作，不修改旧行：

```json
{
  "type": "input_edited",
  "id": "event-2",
  "input_id": "input-1",
  "expected_revision": 1,
  "content": "修改后的问题"
}
```

重放时 revision 不匹配会被视为损坏或冲突事件。

### 4.3 TurnCommitted

一次成功 Turn 使用单个事件保存完整有序消息段：

```json
{
  "type": "turn_committed",
  "id": "event-3",
  "base_leaf_id": "entry-B",
  "consumed_inputs": [{"id": "input-1", "revision": 2}],
  "entries": [
    {"type": "message", "id": "entry-U", "parent_id": "entry-B", "message": {}},
    {"type": "message", "id": "entry-A", "parent_id": "entry-U", "message": {}}
  ],
  "new_leaf_id": "entry-A"
}
```

该事件在一个 JSONL record 中同时完成：

- 消费 pending queue 的连续前缀；
- 把 UserMessage 和本轮所有输出加入 Entry Tree；
- 把 active leaf 移动到最后一个 Entry。

因此不会出现“消息已提交，但 PendingInput 尚未移除”的可见中间状态。

### 4.4 LeafChanged

从过去节点创建或切换分支时追加导航事件：

```json
{
  "type": "leaf_changed",
  "from_leaf_id": "entry-D",
  "target_leaf_id": "entry-B"
}
```

后续 Turn 以 `entry-B` 为 `base_leaf_id`，形成新的子链。原分支仍保存在日志和投影中。

## 5. JSONL 文件

一个 Session 对应一个 UTF-8 `.jsonl` 文件，文件名为 `sha256(session_id)`。第一行是版本化
Header，之后每行是一个完整 SessionEvent（Context Summary 启用后也可出现
`context_summary_created`）：

```jsonl
{"type":"session","version":1,"id":"session-1","created_at":"...","metadata":{}}
{"type":"input_enqueued",...}
{"type":"input_edited",...}
{"type":"turn_committed",...}
{"type":"leaf_changed",...}
```

每次公开 mutation 只有在追加、flush 和 `fsync` 成功后才算持久化。进程崩溃可能留下不完整
的最后一行；加载器只允许截断这一条无换行尾记录。中间损坏、未知事件、重复 Event ID、父链
错误或 revision 冲突都会明确失败，不能静默跳过。

JSONL 首次创建写入 Header 和首批事件。`JsonlSessionStore` 仍缓存 live Session，以保持当前
单进程 Runtime 在 RUN 期间追加输入的语义。

## 6. Runtime SAVE

ExecutionContext 记录：

```python
base_leaf_id: str | None
save_cursor: int  # 当前 active branch 的消息数量
```

PREPARE 使用 `session.active_messages()` 构造 working copy。SAVE 必须确认 active leaf 和历史
前缀没有变化，再调用 `commit_working_messages()` 产生一个 `TurnCommitted`。`failed` 和
`cancelled` 不产生提交事件，已有 `InputEnqueued` 因而继续投影为 PendingInput。

RUN 期间新入队的消息会立即形成自己的 `InputEnqueued`，但本轮 `TurnCommitted` 只消费执行
开始时的队首，后来消息继续留在队尾。

## 7. 快照兼容

`JsonFileSessionStore` 和版本化 `SessionCodec` 继续保留，作为调试、迁移和未来 replay snapshot
的基础。Snapshot schema v6 保存 `entries`、`active_leaf_id`、`context_summaries`、工具结果的
`artifact_refs`/metadata 与强类型 Runtime Status；codec 可以读取 schema v1-v5，其中 v2 不含
Summary，v1 的平铺 messages 会自动
迁移为一条线性 Entry 链。

当前耐久运行的主要实现是 `JsonlSessionStore`。快照不是 JSONL Event Log 的事实来源。

## 8. 当前边界

- 第一版只承诺单进程写入，不包含多进程 lease；
- Session metadata 尚未事件化，JSONL Header 只保存创建时 metadata；
- 没有日志压缩、snapshot checkpoint 或 schema migration 命令；
- 没有工具副作用 exactly-once 或 in-flight Provider stream 恢复；
- Entry Tree 只保存 MessageEntry；模型配置、记忆、ContextSummary 和 ContextCheckpoint 不进入对话树。

## 9. 临时提问窗口

临时提问可以从任意 `base_entry_id` 创建独立 TemporarySession：

```text
主 Session 的某个 Entry
  → 继承根到该 Entry 的上下文
  → 临时 Session 独立追加事件和消息
  → 默认不改变主 Session active leaf
```

关闭时可以丢弃，或显式转成正式 Session。未来也可以把临时窗口实现成同一 Entry Tree 中未被
选为 active leaf 的分支，但产品层必须保持“不会隐式改变主对话”的语义。

## 10. 已确定的后续框架：树外 ContextSummary

历史压缩采用“保留事实树、替换上下文视图”的原则，不删除、不改写旧 MessageEntry，也不把
Summary 作为节点加入 Conversation Entry Tree：

```text
Conversation Entry Tree = 完整、可导航、可分支的真实交互
ContextSummary          = 分支相关的压缩边界与结构化摘要
Context View             = Summary + 当前分支未覆盖的消息尾部
```

Entry Tree 继续只保存 UserMessage、AssistantMessage 和 ToolResultMessage。计划在树外增加：

```python
ContextSummary(
    id=...,
    session_id=...,
    covered_through_entry_id=...,  # Summary 覆盖到当前分支的哪个节点
    source_leaf_id=...,            # 生成 Summary 时的 active leaf，仅用于审计
    content=ContextSummaryContent(...),
    tokens_before=...,
    previous_summary_id=...,
    created_at=...,
)
```

`covered_through_entry_id` 是平铺消息压缩索引的稳定树结构替代。它表示 Summary 已经覆盖从根到
该 Entry 的事实历史；当前 Active Path 中位于该 Entry 之后的消息继续以原文进入模型。

```text
Conversation Tree:
A → B → C → D → E → F

ContextSummary:
summary = Summary(A～C)
covered_through_entry_id = C
source_leaf_id = F

模型 Context View:
Summary(A～C) + D → E → F
```

Summary 不改变 Active Leaf。它适用于所有包含 `covered_through_entry_id` 的后代分支：

```text
A → B → C
        ├─ D → E → F   => Summary(A～C) + D → E → F
        └─ X → Y       => Summary(A～C) + X → Y

A → B → Z              => C 不在路径中，不使用该 Summary
```

多个 Summary 同时适用时，选择当前路径上覆盖位置最深的一个；覆盖边界相同时选择最新版本。
重复压缩通过 `previous_summary_id` 使用“上一次 Summary + 此后新增且再次被覆盖的消息”生成新
Summary，不重新总结完整原始历史。

已经实现非树事件 `ContextSummaryCreated`。它进入同一个 Session Event Log，以保证
Summary 和边界原子耐久化，但不会进入 `Session.entries`，也不会移动 Active Leaf：

```text
Session Event Log 中存在 ≠ Conversation Entry Tree 中存在
```

第一版可以把完整 Summary 存在事件中；未来引入 MemoryStore 或 BlobStore 后，可以把正文外置，
事件只保存稳定引用、哈希和边界信息。

已经实现 `SessionContextResolver`：

```python
ResolvedSessionContext(
    summary: ContextSummary | None,
    messages: tuple[AgentMessage, ...],
    summary_id: str | None,
)
```

Resolver 沿 Active Path 选择 Summary 并裁掉已覆盖前缀；ContextBuilder 把结构化 Summary 作为明确的
`Archived Conversation Context` 系统模块注入，再附加保留的原始消息尾部。Summary 必须按数据
边界包裹，保留历史事实、决定和用户约束，但不能把其中引用的文本提升为新的系统指令。

具体压缩算法和 Runner 请求前的临时上下文治理仍分开设计：前者产生耐久
`ContextSummaryCreated`，后者只生成一次 Provider 请求使用的临时消息视图，不得修改
Session Event Log 或 Conversation Entry Tree。

## 11. 已确定的 v1 压缩算法：nanobot Consolidator 基线

第一版直接采用 nanobot 已验证的 token-budget consolidation 方法，不重新设计阈值和循环策略；
后续根据实际模型与会话数据微调。

### 11.1 Token 预算

```text
input_budget
= context_window_tokens
- max_completion_tokens
- safety_buffer
```

v1 使用 nanobot 的默认安全缓冲：

```text
safety_buffer = 1024 tokens
```

估算由 `PromptTokenEstimatorChain` 完成：优先调用 Provider 可选的同步或异步
`count_prompt_tokens(...)` 能力，失败时使用可选 `tiktoken` 本地 tokenizer，再失败则不执行耐久
压缩，交给 Runner ContextGovernor 的 emergency snip 兜底。普通模型响应里的
`usage.prompt_tokens` 发生在请求成功之后，只用于 usage 观测和累计，不能代替本次请求发送前的
窗口检查。

### 11.2 触发和目标

PREPARE 携带真实 PendingInput 复检时，只有当前完整 Prompt 估算达到或超过 `input_budget`
才触发压缩。SAVE 后还没有下一条真实输入，因此后台探测提前预留一段输入空间：

```text
proactive_input_reserve_tokens = 2048
proactive_trigger_budget
= input_budget - min(proactive_input_reserve_tokens, input_budget - 1)
```

预留只改变后台探测的触发阈值，不伪造 Prompt token 数，也不在下一轮真实输入复检时重复扣除。
对小于默认预留的测试窗口或模型窗口，实际预留自动收敛到 `input_budget - 1`，确保触发预算始终
为正数。触发后不只压到刚好低于上限，而是压到：

```text
target_tokens = input_budget × consolidation_ratio
consolidation_ratio = 0.5
```

后台探测使用 `min(target_tokens, proactive_trigger_budget)` 作为目标；真实输入复检仍使用
`target_tokens`。这样为后续工具循环和新消息保留空间，避免每轮重复触发。

### 11.3 边界选择

Consolidator 只检查当前 Active Path，从最早尚未被最新 Summary 覆盖的消息开始累计待移除
token，并选择第一个已经移除足够 token 的安全 User Turn 边界。

树结构适配：

```text
nanobot end_index                → covered_through_entry_id
session.last_consolidated        → 最新适用 ContextSummary
session.messages[index:]         → Active Path 中边界后的 MessageEntry
```

第一版不主动拆分单个 Turn。边界必须避免让保留上下文从孤立 ToolResult 或未闭合工具调用开始；
如果没有安全边界，本轮不创建 Summary，由 Runner 临时治理保证 Provider 请求可发送。

### 11.4 多轮压缩

每次压缩生成：

```text
previous_summary.content
+ 本次新覆盖的原始消息
→ new_summary
```

最多执行 5 轮 consolidation。每轮成功后重新估算完整 Prompt；达到 target、没有安全边界或生成
失败时停止。

### 11.5 成功与失败语义

Summary 成功后追加单个 `ContextSummaryCreated`，其中保存结构化内容、覆盖边界、生成时 Leaf、
token 估算和 previous summary 引用。

nanobot 在摘要模型失败时会把原文写入 MemoryStore raw archive 并推进索引。当前 Harness 尚无
等价的长期 Memory/BlobStore，因此 v1 不复制这个副作用：失败时不创建 Summary、不推进覆盖
边界，Conversation Entry Tree 保持不变，并停止本次 Consolidator 重试。

### 11.6 调用时机

Runtime 采用 nanobot 的 SAVE 后后台探测与 PREPARE 同步复检：

```text
SAVE（TurnCommitted 已 fsync）
→ schedule background probe
→ RESPOND

下一轮 PREPARE
→ await same-Session background probe
→ recheck with real PendingInput
→ reload and validate Session
→ resolve context
→ ContextBuilder
```

后台探测使用临时 UserMessage probe 和默认 2048-token 输入预留，两者都不进入 Session；
PREPARE 使用真实 PendingInput 且不再叠加预留，因此后台负责隐藏常见摘要延迟，前台复检负责最终
预算正确性。ContextConsolidator 为每个 Session 使用独立锁，Runtime 同时跟踪后台 Task，避免
重复摘要。后台任务只通过 Session ID 重新读取当前聚合，不持有 SAVE 阶段的陈旧 Session 快照。
CLI 清空会先取消对应任务，进程退出会排空剩余任务。

### 11.7 与 Runner ContextGovernor 的关系

耐久 Consolidator 失败、没有安全边界或 token 估算漂移时，Runner 仍需在每次 Provider 调用前
执行纯视图治理：工具链修复、旧工具结果 microcompact、单结果预算和 emergency history snip。
这些变换不得产生 SessionEvent，也不得修改 Conversation Entry Tree 或 ContextSummary。

## 12. ContextSummary 输出与严格解析协议

`ContextSummary` 与用于 Runner 任务恢复的 `ContextCheckpoint` 是两个不同概念：前者保存会进入
模型上下文的压缩语义，后者保存执行阶段和游标等恢复状态。Consolidator 只生成
`ContextSummary.content`；`id`、Session、覆盖边界、来源 Leaf、token 估算和前序 Summary 引用
全部由 Harness 生成。

### 12.1 结构化内容

```python
ContextSummaryContent(
    schema_version=1,
    objective=str | None,
    status="active" | "waiting_for_user" | "blocked" | "completed" | "unclear",
    user_constraints=list[str],
    established_facts=list[str],
    decisions=list[SummaryDecision],
    completed_work=list[str],
    current_work=list[str],
    next_steps=list[str],
    artifacts=list[SummaryArtifact],
    unresolved_questions=list[str],
    known_issues=list[str],
    continuation_note=str | None,
)
```

没有任何待压缩消息时不调用模型。存在待压缩消息但没有值得保留的内容时，仍返回同一完整对象，
其可空字符串字段为 `None`、列表为空，不使用 nanobot 自由文本协议中的 `"(nothing)"` 哨兵。
如果存在前序 Summary，新输出必须是合并新消息后的完整替代版本，不能因本批消息无新增事实而把
前序内容清空。

### 12.2 提示词与输入

System Prompt 要求模型只依据 `previous_summary + new_messages` 生成完整替代对象；新消息冲突时
覆盖旧状态；优先保留用户纠正和约束、决定、当前进度、未决问题、精确标识符、文件路径和错误；
已经解决的事项要移出当前状态；旧的完成项要聚合，避免 Summary 自身无限增长。消息正文和工具
结果都按不可信数据处理，不能改变 Consolidator 指令，也不得持久化凭据、token 或私钥。

输入作为一个 JSON 数据对象放入 User Message，其中包含 `previous_summary` 和带 Entry ID、消息
类型、时间及工具关联信息的 `new_messages`。固定 JSON key 使用英文，自然语言值沿用对话的主要
语言，代码标识符、路径、命令和错误文本不翻译。

### 12.3 Provider 与解析边界

当前 `LLMProvider.complete()` 只承诺返回 `LLMResponse.content: str`，尚未把结构化输出 Schema
纳入 Provider-neutral 接口。v1 因而要求模型把 Summary JSON 写入 `content`，再由 Harness：

```text
LLMResponse.content
→ 严格 JSON 解码
→ Schema 与大小校验
→ ContextSummaryContent
```

只接受一个完整 JSON object；拒绝 Markdown fence、前后说明、重复 key、NaN/Infinity、尾随内容、
缺失或未知字段、错误类型、错误枚举和空列表项，不做字符串/数字等隐式类型转换。成功后只保存
强类型对象，不保存模型的原始输出。规范化 JSON 目标不超过 6000 字符，硬限制 8000 字符；超限
不能截断，必须视为校验失败。

首次语法或 Schema 校验失败时，带原始输入、错误输出和精确字段路径进行一次修复请求。第二次
仍失败则本轮 Consolidator 失败：不产生 `ContextSummaryCreated`，不推进覆盖边界，继续使用上一份
有效 Summary。未来 Provider 原生支持 JSON Schema 时可以在生成阶段增加第一层约束，但 Harness
本地校验始终保留为最终领域边界。
