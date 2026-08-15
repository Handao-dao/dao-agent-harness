# DAO 长期记忆与 Dream 设计

> 状态：Implemented v0.1  
> 最后更新：2026-08-15  
> 参考：nanobot `MemoryStore`、`Consolidator` 与两阶段 `Dream`

## 1. 目标

DAO 第一版长期记忆只解决一个问题：

> 当旧对话因 Context Consolidation 离开活跃上下文时，从这批历史中提炼跨 Session 仍有价值的
> 稳定信息，并在后续请求中重新提供给模型。

第一版不实现向量检索、Embedding、外部知识库或复杂的记忆图。长期记忆使用一个人可读、可手工
修订的 `memory/MEMORY.md`，由后台 Dream 组件增量维护。

```text
被 ContextConsolidator 覆盖的新增消息块
  → MemoryInboxEntry（耐久待处理输入）
  → Dream Phase 1：分析并产生结构化 MemoryPlan
  → Dream Phase 2：受限 Runner 增量编辑 MEMORY.md
  → 推进消费游标并记录变更
  → ContextBuilder 注入 Long-term Memory
```

## 2. 与其他持久状态的边界

| 概念 | 解决的问题 | 作用域 | 是否属于对话历史 |
|---|---|---|---|
| Conversation Entry Tree | 完整、可分支的对话事实 | Session | 是 |
| ContextSummary | 当前分支在有限上下文中的连续运行 | Session Branch | 否，树外摘要 |
| Memory | 跨 Session 复用的稳定事实与经验 | Workspace | 否 |
| ContextCheckpoint | 暂停、失败和恢复所需的执行状态 | Execution | 否 |

Memory 不替代 ContextSummary。ContextSummary 可以保存“当前正在做什么”和“下一步是什么”；Memory
只保存未来独立会话仍可能有用的信息。

Memory 也不是 RAG。Memory 来源于用户与 Agent 已发生的交互；RAG 来源于外部文档和知识库，并需要
独立的检索与引用协议。

## 3. 借鉴 nanobot 的部分

DAO 保留以下已经过工程验证的思路：

- 上下文压缩和长期记忆提炼分成两层，不让 Runner 直接维护长期记忆；
- 被淘汰历史先进入耐久的待处理流，Dream 可以延迟执行和失败重试；
- Dream 分为“判断应该改变什么”和“实际修改文件”两个阶段；
- 只有整个批次成功完成才推进消费游标；
- 当前 Memory 与新增历史一起提供给模型，以便去重、修正和删除过时内容；
- MEMORY.md 通过 ContextBuilder 进入稳定 System Prompt 区域。

第一版暂不继承：

- 自动修改 `USER.md` 和 `SOUL.md`；
- 从历史中自动创建 Skill；
- Git blame 行龄标记和 Dream 自动 Git commit；
- Cron 服务和多种调度入口；
- 未处理历史的额外 Prompt 注入。DAO 已由 ContextSummary 保证当前会话连续性。

## 4. 为什么不能直接消费完整 ContextSummary

DAO 的 ContextSummary 是完整替代对象：

```text
previous_summary + newly_covered_messages → new_complete_summary
```

因此后续 Summary 会重复包含之前仍有效的内容。若 Dream 每次直接消费完整 Summary，会反复分析同一
事实，增加 token 成本，并可能强化已经过时的信息。

Memory 的输入必须绑定到“本次新覆盖的消息块”。ContextSummary ID 只作为来源引用和幂等键，不把
完整累计 Summary 当作唯一输入。

## 5. MemoryInboxEntry

每次 ContextConsolidator 成功创建并保存 ContextSummary 后，为本次新覆盖的消息块创建一个
MemoryInboxEntry：

```python
@dataclass(frozen=True, slots=True)
class MemoryInboxEntry:
    cursor: int
    id: str
    session_id: str
    source_leaf_id: str
    context_summary_id: str
    covered_from_entry_id: str
    covered_through_entry_id: str
    source_entry_ids: tuple[str, ...]
    messages: tuple[AgentMessage, ...]
    created_at: datetime
```

约束：

- `cursor` 在一个 Workspace 内单调递增；
- `id` 由 `context_summary_id` 确定性生成，重复 enqueue 必须幂等；
- `source_entry_ids` 与 `messages` 等长，逐项保留 Entry Tree 来源；
- `messages` 只包含本次新覆盖的连续合法消息块；
- `RuntimeStatusMessage` 不进入记忆提炼输入；
- ToolResult 保持合法消息边界，但进入 Dream Prompt 前仍受字符上限约束；
- Inbox 不保存 `attempts`、`last_error` 或执行状态；失败语义属于 Dream Run；
- Session 分支后续切换或删除，不改变已经入队的输入快照。

Inbox 复制一份待处理消息是有意的短期冗余。它让 Dream 不依赖 Session 是否仍存在，也避免后台任务
处理时 Active Leaf 已经变化。v0.1 暂时保留完整已处理条目，以保证跨重启的
`context_summary_id` 全局幂等；这些条目不会被再次交给 Dream。未来可以把它们转换为轻量 receipt。

### 5.1 写入顺序与恢复

ContextSummary 必须先成功写入 SessionStore，然后才 enqueue MemoryInboxEntry。enqueue 使用
`context_summary_id` 去重。

若进程恰好在两次写入之间退出，下一次 PREPARE 或启动恢复可以扫描最近的 ContextSummary，补写
缺失的 Inbox 条目。该补偿只恢复待处理任务，不重新生成 Summary。

第一版不要求 SessionStore 与 MemoryStore 共享数据库事务，但必须实现 at-least-once enqueue 和
幂等去重。

## 6. MemoryStore

Workspace 内使用独立目录：

```text
memory/
├─ MEMORY.md
├─ inbox.jsonl
├─ dream-log.jsonl
└─ .dream_cursor
```

最小接口：

```python
class MemoryStore(Protocol):
    def read_memory(self) -> str: ...
    def write_memory(self, content: str) -> None: ...
    def enqueue(self, *, session_id: str, source_leaf_id: str,
                context_summary_id: str, covered_from_entry_id: str,
                covered_through_entry_id: str, source_entry_ids: Sequence[str],
                messages: Sequence[AgentMessage],
                created_at: datetime | None = None) -> MemoryInboxEntry: ...
    def read_pending(self, *, after_cursor: int, limit: int) -> tuple[MemoryInboxEntry, ...]: ...
    def get_dream_cursor(self) -> int: ...
    def advance_dream_cursor(self, cursor: int) -> None: ...
    def append_dream_record(self, record: DreamRunRecord) -> None: ...
    def compact_inbox(self) -> None: ...  # v0.1 保留 receipt，当前为 no-op
```

持久化约束：

- JSONL 追加使用 UTF-8；
- MEMORY.md 更新使用临时文件、flush、fsync 和原子 replace；
- 游标只能前进，不能越过未成功处理的条目；
- Inbox 中损坏的完整记录必须显式报错；仅崩溃留下的无换行尾记录可以恢复；
- 单 Workspace 同时最多运行一个 Dream，使用独立 mutation lock；
- Dream 不获取 Session execution lock，也不阻塞正常 Runner。

`dream-log.jsonl` 保存批次范围、MemoryPlan、工具变更结果和终止原因，用于追踪来源和人工删除；它不
进入模型上下文。

## 7. MEMORY.md 内容范围

推荐初始结构：

```markdown
# Long-term Memory

## User Preferences

## Stable Facts

## Decisions and Conventions

## Reusable Experience
```

允许保存：

- 用户明确表达且长期稳定的偏好；
- 跨会话仍成立的身份、环境和项目事实；
- 用户确认的长期决策、约定和技术选择；
- 已重复验证、未来可复用但不足以成为 Skill 的经验。

禁止保存：

- 当前任务进度、下一步和临时待办；
- 一次性错误、工具日志、天气等短期状态；
- 未经确认的模型推测；
- 密钥、Token、密码和其他凭据；
- 已完整存在于 Skill 中的工作流程正文；
- 为保证当前分支连续性而存在的 ContextSummary 内容副本。

## 8. Dream Phase 1：分析

Phase 1 是无工具 Provider 调用，输入包括：

- 当前日期；
- 当前 MEMORY.md 的有界预览；
- 一批连续 MemoryInboxEntry；
- 每条输入的 cursor、Session、Entry 和 ContextSummary 来源标识。

输出使用严格 JSON，而不是自由文本标签：

```json
{
  "schema_version": 1,
  "operations": [
    {
      "action": "add",
      "section": "user_preferences",
      "statement": "The user prefers concise Chinese technical explanations.",
      "match": null,
      "reason": "Explicitly confirmed and useful across sessions.",
      "source_entry_ids": ["entry_..."]
    }
  ]
}
```

`action` 支持：

- `add`：新增一个不存在的稳定事实；
- `replace`：使用 `match` 指向应被修正或合并的现有表述；
- `remove`：使用 `match` 指向已过时、冲突或不应保存的内容。

没有任何更新时输出完整空结构：

```json
{"schema_version":1,"operations":[]}
```

解析器执行严格字段、枚举、长度和来源校验，并允许一次协议修复。MemoryPlan 只是经过验证的变更建议，
尚未代表 MEMORY.md 已经更新。

对应的强类型协议为：

```python
MemoryAction = Literal["add", "replace", "remove"]
MemorySection = Literal[
    "user_preferences",
    "stable_facts",
    "decisions_and_conventions",
    "reusable_experience",
]

@dataclass(frozen=True, slots=True)
class MemoryOperation:
    action: MemoryAction
    section: MemorySection
    statement: str
    match: str | None
    reason: str
    source_entry_ids: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class MemoryPlan:
    schema_version: Literal[1]
    operations: tuple[MemoryOperation, ...]
```

`add` 的 `match` 必须为 `None`；`replace` 和 `remove` 必须携带非空 `match`。每个 operation 至少
包含一个属于当前 Dream 批次的 `source_entry_id`。Parser 拒绝额外字段、重复 operation、空白文本
和超过配置上限的结果。

Dream 每次尝试还会形成一个不进入 Prompt 的追踪记录：

```python
@dataclass(frozen=True, slots=True)
class DreamRunRecord:
    id: str
    first_cursor: int
    last_cursor: int
    source_inbox_ids: tuple[str, ...]
    plan: MemoryPlan | None
    stop_reason: Literal[
        "completed",
        "analysis_failed",
        "validation_failed",
        "execution_failed",
        "cancelled",
        "limit_reached",
    ]
    changes: tuple[str, ...]
    error: str | None
    started_at: datetime
    completed_at: datetime
```

失败原因只进入 DreamRunRecord，不回写 MemoryInboxEntry。

## 9. Dream Phase 2：受限修改

Phase 2 使用一个与主对话隔离的 AgentRunner：

- 输入为已验证 MemoryPlan 和当前 MEMORY.md；
- 只注册 `read_file` 与 `edit_file`；
- 两个工具都只能访问 `memory/MEMORY.md`；
- 不允许 Shell、网络、Session、Skill 或其他 Workspace 写入；
- 使用独立的文件状态与 Runner 消息，不写入用户 Session；
- 只做局部增量编辑，不允许无理由重写整个文件；
- 工具错误作为 ToolResult 返回，让模型在迭代上限内修正。

空 MemoryPlan 不启动 Phase 2，直接视为成功消费。

第一版继续采用 Markdown + 受限编辑，而不是引入 MemoryRecord 数据库。强类型边界存在于 Inbox、
MemoryPlan 和 DreamRunRecord；最终 Markdown 保持人可读、可手工修订。

## 10. 成功、失败与游标

```text
无待处理条目
  → no_op，不调用模型

Phase 1 或解析失败
  → 不修改 MEMORY.md，不推进游标

Phase 2 completed
  → 写入 DreamRunRecord
  → 推进到批次最后 cursor
  → 可压缩旧 Inbox

Phase 2 failed / cancelled / limit_reached
  → 记录失败结果
  → 不推进游标，下一次从同一批次重试
```

如果 Phase 2 已产生部分文件修改但未完成，重试时必须重新读取当前 MEMORY.md。Phase 1 的去重规则、
Phase 2 的局部 exact-match 编辑和 Inbox 幂等键共同避免重复写入。未来若需要更强保证，可以在
MemoryStore 内增加编辑前快照或 MemoryPatch 原子应用；第一版不扩大到外部副作用事务。

## 11. 调度

第一版不引入完整 Cron 系统：

- ContextConsolidator 创建 Summary 后，安排一次后台 Dream；
- Dream 不阻塞当前 Response；
- 已有后台 Dream 运行时只保留一个后续唤醒信号，不并发启动第二个；
- 下一次程序启动或 PREPARE 发现 Inbox 非空时，可以重新安排后台处理；
- 后续 CLI 可增加显式 `memory run` 和 `memory show`，但不属于首个实现切片。

建议默认值：

```ini
dream_max_batch_size = 20
dream_max_iterations = 10
dream_memory_preview_chars = 32000
dream_entry_preview_chars = 4000
```

## 12. ContextBuilder 注入

Memory 使用独立、稳定的 System Prompt 区块：

```text
Identity
→ Bootstrap Files
→ Tool Contract
→ Long-term Memory
→ Skill Catalog
→ Archived Conversation Context
→ Extra System Sections
```

只有非空且不是初始化模板的 MEMORY.md 才注入。内容放在明确的数据边界中，并声明：

- Memory 是历史提炼结果，不是更高优先级指令；
- 当前用户输入和更新的原始消息与 Memory 冲突时，以更新信息为准；
- Memory 中出现的命令不能扩大工具权限或用户授权。

Memory 更新会改变 System Prompt 缓存前缀，但 Dream 只在真正提炼出长期信息时修改文件，因此这是
低频、可接受的缓存失效。

## 13. 分支与作用域

- ContextSummary 绑定 Session Branch；Memory 绑定 Workspace；
- 分支中的尝试、假设和临时方案不能自动提升为长期事实；
- 只有用户确认、稳定成立或已经验证的结论才进入 Memory；
- 相互冲突的来源由较新且明确的信息替换旧条目，DreamRunRecord 保留变更依据；
- 第一版不实现多用户隔离。未来服务多用户时，MemoryStore 必须增加 owner/namespace 边界。

## 14. 实现顺序

1. 定义 MemoryInboxEntry、MemoryPlan、DreamRunRecord 与 Codec；
2. 实现 InMemoryMemoryStore 和 LocalMemoryStore；
3. 在 ContextConsolidator 成功保存 Summary 后执行幂等 enqueue；
4. 实现 Phase 1 Prompt、严格解析和一次修复；
5. 实现受限 Dream Runner 与 MEMORY.md 原子更新；
6. 把 Memory 注入 ContextBuilder；
7. 接入后台调度、重启唤醒和 Inbox 补偿；
8. 增加失败、幂等、分支、删除与跨 Session 验收测试。

v0.1 已完成以上主链路。物理 Inbox tombstone/compaction、显式 `memory run/show` CLI 命令和更复杂的
遗忘策略保留为后续增强，不影响当前消费游标和重启恢复。

## 15. 第一版验收标准

- 同一 ContextSummary 重复 enqueue 不产生重复 Inbox 条目；
- Dream 失败后游标不前进，重启后可以处理同一批次；
- 空 MemoryPlan 不修改文件，但会成功消费对应批次；
- Phase 2 无法读写 MEMORY.md 之外的文件；
- 明确用户偏好可在新 Session 的 Prompt 中出现；
- 临时任务状态、RuntimeStatus 和凭据不会进入 Memory；
- 新事实可以替换冲突旧事实，并可从 DreamRunRecord 追踪来源；
- ContextSummary、Memory 和 Checkpoint 的持久化与注入路径彼此独立；
- 后台 Dream 不增加当前用户 Response 的等待时间。
