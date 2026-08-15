# DAO Agent Harness 实施状态与提取清单

> 状态：Living Document v0.7  
> 最后更新：2026-08-14  
> 用途：记录 nanobot 参考来源、当前真实完成度和后续路线

## 1. 演进原则

- nanobot 是循环行为和工程经验参考，不要求复制全部产品能力；
- pi agent-core 用于对照 Provider、消息和工具边界；
- 先建立可运行且可测试的主链路，再按实际需要增加横向能力；
- 已暂缓能力不能出现在当前目录结构或当前接口说明中；
- 每次改变消息、保存或工具协议，都必须用不变量测试保护；
- 新实现保持独立，不从 `agent_harness` 导入 nanobot 包。
- 不为压缩行数拆散领域语义；只删除不承载协议、不变量或复用价值的机械抽象。

清单标记：`[x]` 表示当前代码和测试均已落地，`[ ]` 表示尚未实现。

## 2. nanobot 参考位置

相对于 `agent-harness` 项目根目录：

| 能力 | nanobot 文件 |
|---|---|
| Runner 主循环与工具批次 | `../nanobot/nanobot/agent/runner.py` |
| 外层阶段状态机 | `../nanobot/nanobot/agent/loop.py` |
| ToolRegistry 参数准备 | `../nanobot/nanobot/agent/tools/registry.py` |
| Tool schema 与并发属性 | `../nanobot/nanobot/agent/tools/base.py` |
| Provider 抽象 | `../nanobot/nanobot/providers/base.py` |
| Session 管理 | `../nanobot/nanobot/session/manager.py` |
| Token 压缩与摘要 | `../nanobot/nanobot/agent/memory.py` |
| Inbound / Outbound 消息 | `../nanobot/nanobot/bus/events.py` |

当前没有复制 nanobot 的 Goal、Subagent、WebUI、Channel、Heartbeat、Cron 或产品特定
安全文案。长期记忆只借鉴 nanobot 的两阶段 Dream 行为，使用 DAO 自有的强类型 Inbox、严格
MemoryPlan、隔离 Runner 和 ContextBuilder 边界重新实现。

## 3. 当前目录结构

```text
src/agent_harness/
├─ __init__.py
├─ __main__.py
├─ artifacts/
├─ checkpoints/
├─ cli.py
├─ context.py
├─ context_governor.py
├─ consolidation.py
├─ messages.py
├─ runner.py
├─ runtime.py
├─ runtime_io.py
├─ session.py
├─ status.py
├─ status_builder.py
├─ summary.py
├─ token_estimation.py
├─ providers/
│  ├─ base.py
│  ├─ fake.py
│  ├─ openai_compatible.py
│  └─ transport.py
├─ storage/
│  ├─ base.py
│  ├─ codec.py
│  ├─ event_codec.py
│  ├─ json_file.py
│  ├─ jsonl.py
│  └─ memory.py
├─ tools/
│  ├─ base.py
│  ├─ builtin.py
│  ├─ fake.py
│  └─ registry.py
└─ templates/
   ├─ archived_context.md
   ├─ context_summary.md
   ├─ identity.md
   └─ tool_contract.md
```

`injection.py` 已经承载运行中消息注入协议；`events.py`、独立 ToolRuntime 和
TransportMessage 当前仍不存在，它们是候选扩展，不是遗漏的空壳组件。

## 4. 已完成：项目与 Provider 基线

- [x] Python 3.11+ `pyproject.toml`；
- [x] pytest asyncio 配置与 Ruff；
- [x] MIT License 和第三方来源说明；
- [x] Provider Protocol；
- [x] Provider-neutral `LLMResponse`；
- [x] `TextDelta`、`ToolCallCompleted`、`ResponseCompleted` 流事件；
- [x] OpenAI-compatible Chat Completions；
- [x] stdlib JSON HTTP 与 SSE transport；
- [x] `agent_harness.testing` 中的 ScriptedProvider 与 FakeTool 测试替身；
- [x] 非流式与流式工具调用参数归一化；
- [x] 流式接口缺失时回退到 complete。

## 5. 已完成：领域消息和 Session

- [x] `UserMessage`、`AssistantMessage`、`ToolResultMessage`、`RuntimeStatusMessage`；
- [x] `ToolCall` 和稳定 `tool_call_id` 关联；
- [x] 每条 AgentMessage 有稳定 ID 和 timestamp；
- [x] Provider dict 不进入 Runner 结果或 Session；
- [x] `PendingInput.id` 同时成为提交后的 UserMessage.id；
- [x] PendingInput 入队和仍在队列时的 source message 去重；
- [x] PendingInput 编辑不修改正式历史；
- [x] Message Entry Tree、parent_id 和 active leaf；
- [x] Session 当前分支复制、base_leaf_id 与 save_cursor；
- [x] working history 前缀、队列前缀和消息 ID 校验；
- [x] 迟到 execution 不能提交已被编辑的 PendingInput；
- [x] `InputEnqueued`、`InputEdited`、`TurnCommitted`、`LeafChanged`；
- [x] 树外 `ContextSummaryCreated` 和分支覆盖边界；
- [x] 完整有序消息段使用单个 TurnCommitted 原子提交；
- [x] 过去节点 checkout 与分支上下文继承；
- [x] SessionStore Protocol；
- [x] InMemorySessionStore、JsonFileSessionStore 和 JsonlSessionStore。

当前 AgentMessage 只支持文本 content。多模态 block、thinking block 和非消息 Entry 尚未实现。

## 6. 已完成：ContextBuilder

- [x] 内置 Identity 和 Tool Contract；
- [x] 按固定顺序读取 `AGENTS.md`、`SOUL.md`、`USER.md`；
- [x] Bootstrap 文件严格 UTF-8；
- [x] 额外 system section 扩展槽；
- [x] AgentMessage 到 Provider-neutral dict 的独立投影；
- [x] 工具参数紧凑 JSON 编码；
- [x] `is_error=True` 工具结果的模型可见错误标记；
- [x] 构建过程不修改 working messages 或 Session。
- [x] 结构化 ContextSummary 的 guarded system section；
- [x] SessionContextResolver 选择最深、最新的适用 Summary；

Memory 和 MCP 尚未实现。Runtime Status、SkillCatalog、Skill 工具与压缩后恢复已实现；
Runner 临时裁剪由 ContextGovernor 负责，也已经实现。

## 7. 已完成：AgentRunner

- [x] typed `AgentRunSpec` 和 `AgentRunResult`；
- [x] 模型—工具多轮循环；
- [x] Provider 异常与 error finish reason；
- [x] 空正常响应的框架兜底消息；
- [x] max_turns 的闭合消息和 `limit_reached`；
- [x] usage 累计；
- [x] 流式文本回调和完整工具调用聚合；
- [x] 失败路径不伪造 AssistantMessage；
- [x] cancelled Run 不保存不完整工具结果；
- [x] Runner 不读取 SessionStore、Channel 或 MessageBus。

## 8. 已完成：工具注册、校验和执行

- [x] `parallel_safe` / `sequential` 工具标签；
- [x] ToolRegistry 注册、重复名称和 schema 边界校验；
- [x] `prepare_call()` 工具解析与 schema 驱动参数转换；
- [x] 常用 JSON Schema 子集校验；
- [x] 连续并行安全调用分组；
- [x] sequential 工具作为独立屏障；
- [x] 并行结果按完成顺序产生；
- [x] 结果通过 `tool_call_id` 关联，不依赖位置；
- [x] 未知工具、参数错误、普通异常和工具主动错误回流模型；
- [x] `asyncio.CancelledError` 不包装成普通工具错误；
- [x] 并行任务异常或取消时回收同批 Task。

当前设计明确不创建独立 ToolRuntime。详见
[工具注册、校验与执行设计](tool-execution-design.md)。

## 9. 已完成：AgentRuntime

- [x] `LOAD → PREPARE → RUN → SAVE → RESPOND → DONE`；
- [x] 显式转换表；
- [x] 用户输入在 Runner 启动前进入 PendingInput；
- [x] system prompt 在 PREPARE 构建一次；
- [x] 每 Session execution lock；
- [x] 不同 Session 并行；
- [x] RUN 期间允许新输入进入队尾；
- [x] SAVE 重新读取最新 Session；
- [x] completed 与 limit_reached 提交 working tail；
- [x] failed 与 cancelled 保留 PendingInput；
- [x] SAVE 冲突不继续 RESPOND；
- [x] RuntimeResult 对外隐藏 Session 和 ExecutionContext。
- [x] SAVE 后调度后台 ContextConsolidator，PREPARE 等待并使用真实输入复检；
- [x] 单条 PendingInput Token 上限、保留修订和运行中注入拦截；
- [x] Runner 只发送 Summary 未覆盖的消息尾部，SAVE 仍校验完整 working history。

## 10. 已完成：ContextSummary 与耐久压缩

- [x] `ContextSummaryContent`、Decision、Artifact 和状态枚举；
- [x] 固定 Consolidator Prompt 和 JSON 数据输入；
- [x] 严格 JSON、重复 key、字段、类型、枚举和大小校验；
- [x] 一次结构修复请求，二次失败不推进覆盖边界；
- [x] Snapshot schema v6、ArtifactRef/Runtime Status 持久化与 JSONL
  `ContextSummaryCreated` replay；
- [x] ContextSummary 与任务恢复 ContextCheckpoint 的命名分离；
- [x] `context_window - completion - 1024` 预算公式和 50% 目标；
- [x] SAVE 后探测默认预留 2048 input tokens，PREPARE 真实输入复检不重复扣除；
- [x] User Turn 安全边界和最多 5 轮压缩；
- [x] 支持同步/异步结果的模型级 PromptTokenEstimator；
- [x] Provider `count_prompt_tokens` → 可选 `tiktoken` 的默认 estimator chain；
- [x] 响应 usage 与请求前 token estimation 的职责分离；
- [x] 估算或生成失败时保持旧 Summary 和原始 Entry Tree。

## 11. 当前验证基线

当前测试覆盖 250 个行为，包括：

- Provider JSON/SSE 翻译；
- Runner 正常、错误、流式、取消和循环限制；
- 工具 schema、错误回流、timeout、并行批次和串行屏障；
- AgentMessage 与 ContextBuilder 不变量；
- PendingInput、Session commit 和迟到 execution 冲突；
- Runtime SAVE 判定与 per-Session 并发；
- Entry Tree 分支、SessionEvent replay、JSONL 尾部恢复。
- ContextSummary 严格解析、修复、分支继承、持久化和 Runtime 尾部投影；
- ContextGovernor、ArtifactStore、Codec v6、Runtime Status 和 CLI 分页回读。
- ContextCheckpoint 三节点、独立 Store、冲突检测和 SAVE 失败恢复。
- MessageInjectionPoint、五条总配额、同批模型视图合并和剩余输入新 Runner 交接。

验证命令：

```bash
uv run --extra dev pytest -p no:cacheprovider
uv run --extra dev ruff check .
```

## 12. 近期可选路线

这些能力会把完整原型推进到可作为应用入口使用，但不阻碍当前最小 Harness 成立。

### 12.1 Runtime 入口接线

- [x] 确定 v1 使用更薄的 CLI Adapter，不引入通用 InboundMessage / OutboundMessage；
- [x] 定义 RuntimeRequest、RuntimeStreamEvent 和 RuntimeResult 的对外边界；
- [x] CLI 调用 `AgentRuntime.submit()`，不再自行维护 history；
- [x] 区分临时流式展示与 SAVE 后 RuntimeResult 正式确认；
- [x] 增加两轮对话、失败后重试和队列推进端到端测试。

### 12.2 真实 Session 持久化

- [x] 原子 JSON Snapshot SessionStore；
- [x] 追加式 JSONL SessionEvent Store；
- [x] 严格 UTF-8、版本 Header 和强类型 Codec；
- [x] 进程重启后 replay 正式分支、Active Leaf 和 PendingInput；
- [x] 不完整最终 JSONL record 的安全截断；
- [ ] SQLite SessionStore；
- [ ] 日志 snapshot、压缩和 migration 命令；
- [ ] 明确多进程并发和 lease 边界。

### 12.3 工具执行增强

- [x] 定义 ToolExecutionPolicy、ToolExecutionResult 和 Registry 单调用执行边界；
- [x] 实现工具级 timeout 与统一结果归一化；
- [x] 完成大型工具结果 ArtifactStore v0.1 协议设计；
- [x] 实现内容寻址的 InMemory/Local ArtifactStore；
- [x] 扩展 ToolExecutionResult、ToolResultMessage 和 Runner 引用传递；
- [x] 升级 Session Codec v6，并兼容读取 v1-v5 旧消息；
- [x] 接入 Registry 成功大结果外置、头尾预览和存储失败降级；
- [x] 实现受控分页读取的 `read_artifact` 内建工具；
- [x] CLI 配置 LocalArtifactStore，并自动注册读取工具；
- [x] ToolOutput 分离模型视图、完整 Artifact 内容和显式错误语义；
- [x] 定义工具专属模型视图优先、未知工具通用 Artifact fallback 的优先级；
- [ ] 为文件、Shell、搜索等具体工具实现结构化模型视图或确定性摘要；
- [ ] approval；
- [ ] sandbox / remote executor；
- [ ] 幂等键和外部副作用恢复；
- [ ] 完整 JSON Schema Draft 支持。

## 13. 已实现：Runner Context Governance

- [x] 耐久压缩 token budget；
- [x] 定义 ContextGovernor 强类型协议、治理顺序和 current Turn 保护边界；
- [x] 实现 orphan ToolResult 删除与 missing ToolResult 补齐；
- [x] 实现 tool result budget、micro-compaction 和 emergency snip；
- [x] 将治理收口为 NORMAL、PRESSURE、EMERGENCY 三级按需降级；
- [x] Emergency 保留任务锚点和最近合法尾部，并允许裁剪当前 Turn 中间轨迹；
- [x] 持久化 ContextSummary compaction；
- [x] SkillCatalog section、激活结果保护与压缩后恢复；
- [x] Runtime Status 强类型快照、逐模型决策注入、压力治理与摘要过滤；
- [ ] Memory 和 MCP sections。

这些转换必须只产生模型视图，不能修改 Session 正式历史或 save_cursor。

## 14. 已实现：ContextCheckpoint 与消息插入

Checkpoint 固定使用三个节点：

- [x] `awaiting_tools`；
- [x] `tools_completed`；
- [x] `final_response`；
- [x] ContextCheckpoint 与 RunnerCheckpoint 强类型；
- [x] InMemoryCheckpointStore 与 JsonFileCheckpointStore；
- [x] 状态未知工具不自动重放；
- [x] PendingInput revision、Active Leaf 和 save_cursor 冲突检测；
- [x] final_response 后 Session SAVE 失败可直接重试提交；
- [x] CLI `--checkpoint-dir` 与 `/clear` 联动。

安全点消息插入固定使用两个检查位置：

- [x] 整批工具完成、下一次模型调用之前；
- [x] 候选最终回答完成、Run 正式结束之前；
- [x] 同批连续 UserMessage 只合并 Provider 模型视图；
- [x] 单 Runner 最多吸收五条追加 PendingInput；
- [x] 最后一次 model iteration 不领取追加消息；
- [x] SAVE 只消费模型已见输入；
- [x] 剩余 PendingInput 由新的 Runner 处理；
- [x] ContextCheckpoint schema v2 记录多输入 `id + revision` 前缀。

恢复不得自动重跑状态未知的工具调用；消息插入通过每个 Runner 的五条总配额限制运行长度。
Checkpoint 的强类型结构、独立 Store、恢复转换与崩溃一致性见
[ContextCheckpoint 设计](checkpoint-design.md)，注入时序与预算见
[消息插入设计](message-injection-design.md)。

## 15. 已实现：主动暂停与局部修订

- [x] 每 Session ActiveRun 与 paused latch；
- [x] 默认选择最新追加 PendingInput，允许显式指定修订目标；
- [x] 通过 UserMessage.id 定位并截断目标及其派生消息；
- [x] 保留目标前模型进度，按保留 AssistantMessage 重算 iteration；
- [x] PendingInput revision 持久化后重新构造 UserMessage；
- [x] 等待 execution lock 的 submit 在暂停状态下不得自动启动；
- [x] 截断工具调用形成 `completed/uncertain` 副作用提示；
- [x] CLI `/pause`、`/edit`、直接替换文本与 `/resume`；
- [ ] 跨进程持久化暂停工作前缀；
- [ ] mutating 工具确认、幂等键和补偿事务。

详细规则见 [主动暂停与消息修订设计](pause-revision-design.md)。

## 16. 已记录但暂缓：事件与观测

- [ ] 持久化 Run Timeline；
- [ ] `RunStarted`、模型请求、工具执行和终止事件；
- [ ] 严格递增的 Run event sequence；
- [ ] 非持久化 StreamEvent；
- [ ] 结构化 logger，而不是单独 DiagnosticEvent；
- [ ] trace 和阶段耗时。

这些事件未来属于运行追踪和恢复，不进入 AgentMessage 或 Session 对话历史。

## 17. 明确非目标

- exactly-once 外部副作用；
- 工具补偿事务；
- 分布式 Session 调度；
- 多 Agent DAG；
- sustained goal；
- Cron / Heartbeat；
- 插件市场；
- WebUI；
- 完整 OpenTelemetry；
- 任意运行事件点 fork / replay。

除非具体应用提出需求，这些能力不进入近期 Harness 主链路。

## 18. 完成度定义

### 最小 Loop：已完成

- [x] 模型调用、工具执行、继续循环和终止；
- [x] Provider、工具和取消错误路径；
- [x] typed working messages；
- [x] 并行与 sequential 工具批次。

### 最小 Harness Core：已完成

- [x] PendingInput 先保存语义；
- [x] Session active branch copy / Entry Tree / TurnCommitted；
- [x] ContextBuilder；
- [x] ContextSummary、分支 Resolver 与耐久 Consolidator；
- [x] AgentRuntime 状态机与 execution lock。

### 应用可用入口：已完成 CLI v1

- [x] CLI 接入 AgentRuntime；
- [x] 进程重启后仍可恢复的 SessionStore。
- [x] 运行中消息追加、主动暂停和 PendingInput 局部修订。

### 生产级运行：部分完成

- [x] 三节点 checkpoint 与恢复；
- [x] 耐久 ContextSummary 压缩；
- [x] Runner 临时 Context Governance；
- [ ] approval、sandbox 与外部副作用治理；
- [ ] 观测和多进程协调。
