# DAO Agent Harness

一套以 nanobot 为行为与代码参考、以 pi agent-core 为边界对照，逐步设计和实现的轻量 Agent
Harness。DAO 是项目的正式产品名称；当前 Python 包名仍为 `agent_harness`。

当前阶段聚焦最小可运行主链路，包括：

- 模型与工具调用循环
- 强类型 AgentMessage
- PendingInput 与 Session 历史
- ContextBuilder
- 结构化 ContextSummary 与 token-budget Consolidator
- Workspace 长期记忆、强类型 Memory Inbox 与两阶段 Dream
- SkillCatalog、渐进式 Skill 激活与资源读取
- 模型可见 Runtime Status 快照
- 六阶段 AgentRuntime
- Provider-neutral 流式事件

项目采用“先提取可运行的 nanobot 精简内核，再逐个优化组件”的演进方式。pi
agent 作为消息模型、Provider 和工具执行设计的对照参考。

## 当前里程碑

已实现最小 Runner 纵向链路：

```text
UserMessage
→ ScriptedProvider
→ AssistantMessage(tool call)
→ FakeTool
→ ToolResult
→ ScriptedProvider
→ Final AssistantMessage
→ AgentRunResult
```

Runner、Runtime 和 Session 已统一使用强类型 AgentMessage。只有 ContextBuilder 到
Provider 的边界使用独立 dict 视图。项目同时提供 OpenAI-compatible
`/chat/completions` Provider、交互式 CLI、UTC 时间工具和一组 workspace 基础编码工具。

Provider 已支持 SSE 流式协议，并对外输出三种与厂商无关的事件：

- `TextDelta`：可立即展示的文本增量
- `ToolCallCompleted`：已完成 JSON 聚合、可以安全执行的工具调用
- `ResponseCompleted`：结束原因与 token usage

OpenAI 的 chunk、tool call index 和 arguments 分片只存在于 Provider 内部。Runner 可以
选择消费这些统一事件，将它们重新聚合成原有 `LLMResponse` 后继续走同一套工具循环；
文本增量通过可选回调交给 CLI 或其他上层应用。Provider 不支持流式接口时，Runner 会
自动回退到 `complete()`。任务恢复 checkpoint、消息插入和 Runner 临时上下文治理均已实现；
耐久 ContextSummary 压缩也已经接入 Runtime PREPARE。

工具执行采用 nanobot 式边界：工具用 `parallel_safe` 或 `sequential` 标签声明并发能力，
ToolRegistry 负责参数转换与 JSON Schema 校验，Runner 把连续并行安全调用组成批次，并将
未知工具、无效参数、工具异常和工具主动错误包装成 ToolResult 交回模型继续判断。统一的
Registry 单调用执行、工具级 timeout 和 ToolExecutionResult 已实现。
大型工具结果将通过内容寻址的 ArtifactStore 外置，Session 只保存稳定引用与头尾预览，并由
`read_artifact` 工具按需分页回读。ArtifactRef、ArtifactPolicy、InMemoryArtifactStore 和
LocalArtifactStore 已实现；Artifact 引用也已贯通 ToolExecutionResult、ToolResultMessage、Runner
和 Session Codec v6。Registry 已能自动外置成功的大型结果，并对存储失败执行受限降级；
CLI 会自动注册 `read_artifact`，供模型按 offset/limit 分页回读原始内容。
ToolOutput 进一步把工具主动设计的模型视图与完整 `artifact_content` 分离，并使用显式
`is_error` 表达领域失败；完整失败日志也可以保存为 Artifact，未适配工具仍由 Registry 执行通用
外置和头尾预览。基础文件工具已经实现共享 workspace 路径策略、完整行/UTF-8 字节截断、文件级
Mutation Queue，以及 pi 风格的 `read/ls/find/grep/write/edit/bash` 七个工具；CLI 默认注册这些工具并
使用 `--workspace` 限制文件访问范围。当前版本定位为 Harness 设计骨架，尚未加入 approval、
sandbox、完整 `.gitignore` 或生产级 Shell 进程树治理。

SkillCatalog 默认发现 `<workspace>/.dao/skills` 和用户级 `.dao/skills`。ContextBuilder 只把
Skill 名称与描述放入系统提示词；模型通过 `activate_skill` 按需加载完整 SKILL.md，再使用
`read_skill_resource` 分段读取引用文档。激活结果仍是标准 ToolResult，但其内部 metadata 会让
ContextGovernor 跳过普通裁剪，并在 ContextSummary 覆盖原调用后重新注入合法工具消息块。
Skill 的 frontmatter、正文与 scripts/references/assets 分层规则见
[DAO Skill Authoring Guide](docs/skill-authoring-guide.md)。

Runtime 已实现 `awaiting_tools`、`tools_completed` 和 `final_response` 三节点 ContextCheckpoint。
CLI 默认使用独立原子 JSON CheckpointStore；恢复不会重放状态未知的工具，最终回答在 Session SAVE
失败后也可以直接重试提交而无需再次调用模型。

运行中追加消息继续复用耐久 PendingInput。Runner 在工具完成后和候选回答后检查新增队列后缀，
同批 UserMessage 只在 Provider 视图中合并；一个 Runner 最多吸收五条追加输入，且最后一次 model
iteration 不再领取消息。SAVE 只消费至少进入过一次 Provider 请求的输入，剩余队列由新的有界
Runner 处理。

消息与 Session 领域层已经实现：

- `UserMessage`、`AssistantMessage`、`ToolResultMessage` 和 `ToolCall`；
- 在 Runner 启动前写入 SessionStore 的 `PendingInput` 队列；
- Message Entry Tree、Active Leaf、PendingInput 编辑和分支安全提交；
- `SessionStore` Protocol、内存 Store、原子 JSON Snapshot Store 和 JSONL Event Store。

`JsonlSessionStore` 把 `InputEnqueued`、`InputEdited`、`TurnCommitted`、`LeafChanged` 和
`ContextSummaryCreated`
追加为强类型事件，进程重启后通过 replay 恢复 Pending Queue、Message Entry Tree 和 Active
Leaf。一次成功 Turn 的完整有序消息段在一个 JSONL record 中提交。

ContextBuilder 按固定顺序组装 Identity、Workspace Bootstrap Files、Tool Contract、可用 Skill
目录和可选的 Archived Conversation Context，并把强类型 AgentMessage 转换为独立模型视图。ContextConsolidator
复用 nanobot 的 token 预算、50% 目标、User Turn 边界和最多 5 轮策略；摘要以强类型
ContextSummary 存在 Entry Tree 外，并通过严格 JSON 校验和一次修复保证边界。
请求前 token 估算优先使用 Provider 的可选 `count_prompt_tokens` 能力，再回退到本地
`tiktoken`；模型响应返回的 usage 继续作为请求后的实际用量记录。
Runner ContextGovernor 使用三级上下文兜底：NORMAL 只做协议修复和单结果裁剪，真实超限后才在
PRESSURE 中压缩旧工具结果，最后由 EMERGENCY 保留任务锚点、最新用户补充和最近合法消息尾部。
所有变换只作用于单次 Provider 视图，不修改 working messages 或 Session。
Runner 还可在每次真实模型决策前追加强类型 RuntimeStatusMessage。CLI 默认启用该能力，向模型
提供当前时间、重复工具异常和 ContextGovernor 隐藏的信息；状态在 Provider 边界映射为标准
user role，成功 SAVE 时随 TurnCommitted 持久化，长期 ContextSummary 会显式忽略过期状态。
成功 SAVE 后 Runtime 会在后台提前探测并生成必要摘要；下一轮 PREPARE 等待该任务并使用真实
PendingInput 复检，从而把通常的压缩延迟隐藏在两轮交互之间。
后台探测默认通过 `proactive_input_reserve_tokens = 2048` 为下一条输入提前留出空间；CLI 可用
`--proactive-input-reserve-tokens` 或环境变量
`AGENT_HARNESS_PROACTIVE_INPUT_RESERVE_TOKENS` 覆盖。真实 PendingInput 复检不会重复扣除该预留。
每次新 ContextSummary 持久化后，其新增覆盖消息块会幂等进入 Memory Inbox。后台 Dream 先生成
严格 JSON MemoryPlan，再通过仅能访问临时 `MEMORY.md` 的隔离 Runner 做局部编辑；成功后才推进
游标。长期记忆属于 Workspace，并由 ContextBuilder 作为低频稳定 System Prompt 区块注入。
CLI 启用 `--context-window-tokens` 后，单条输入默认不得超过可用输入预算的一半；可通过
`--max-input-tokens` 显式覆盖。超限消息保留在 Pending Queue，允许编辑后重试，不会进入 Provider。

AgentRuntime 已实现 `LOAD → PREPARE → RUN → SAVE → RESPOND → DONE`。用户输入先持久化到
PendingInput；`completed` 和具备框架终止消息的 `limit_reached` 会提交正式历史，Provider
失败或取消则保留 PendingInput。CLI 已通过强类型 RuntimeRequest 和分段 RuntimeStreamEvent
接入 Runtime，并使用 JSONL Session Event 持久化多轮对话。

## 使用

PowerShell：

```powershell
$env:AGENT_HARNESS_MODEL = "your-model"
$env:AGENT_HARNESS_API_KEY = "your-api-key"
uv run agent-harness
```

连接本地 OpenAI-compatible 服务：

```powershell
uv run agent-harness --model your-model --base-url http://localhost:8000/v1
```

CLI 当前支持流式文本输出、运行中消息追加，以及 `/pause`、`/edit <text>`、`/resume`、
`/retry`、`/clear` 和 `/exit`。暂停默认修订最新追加消息；如果没有追加消息，则修订本轮初始
输入。默认 Session 数据保存在
`~/.dao-agent/sessions`，可通过 `--session-dir` 和 `--session-id` 指定其他位置或对话。
Artifact 默认保存在 `~/.dao-agent/artifacts`，可通过 `--artifact-dir` 覆盖。
`--tool-timeout` 配置 Registry 默认工具 deadline，工具自身可以声明更具体的 timeout。
通过 `--context-window-tokens` 可以启用 Runner ContextGovernor；DAO 不在缺少模型规格时猜测
上下文窗口。

## 开发

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

启用本地 tokenizer fallback：

```bash
uv sync --extra tokenizers
```

当前验证基线为 276 个测试，Ruff 全量检查通过。

## 文档

- [产品愿景、简历项目与实施路线](docs/product-vision-resume-roadmap.md)
- [第一阶段验收](docs/stage-1-acceptance.md)
- [ContextCheckpoint 设计](docs/checkpoint-design.md)
- [消息插入设计](docs/message-injection-design.md)
- [主动暂停与消息修订设计](docs/pause-revision-design.md)
- [组件设计总览](docs/component-overview.md)
- [ContextBuilder 设计](docs/context-builder-design.md)
- [Runner 设计](docs/runner-design.md)
- [工具注册、校验与执行设计](docs/tool-execution-design.md)
- [基础工具设计](docs/basic-tools-design.md)
- [ArtifactStore 设计](docs/artifact-store-design.md)
- [Skill 系统设计](docs/skill-system-design.md)
- [Agent Runtime、Session 与 ContextBuilder 设计](docs/runtime-design.md)
- [CLI 与 Runtime 对外协议](docs/cli-runtime-protocol.md)
- [Runner ContextGovernor 设计](docs/context-governor-design.md)
- [Runtime Status 设计](docs/runtime-status-design.md)
- [Session Entry Tree 与事件持久化设计](docs/session-persistence-design.md)
- [长期记忆与 Dream 设计](docs/memory-system-design.md)
- [实施状态与提取清单](docs/extraction-checklist.md)
