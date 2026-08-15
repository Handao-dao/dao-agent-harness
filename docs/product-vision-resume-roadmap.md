# DAO Agent Harness：产品愿景、简历项目与实施路线

> 状态：Living Document v0.1  
> 最后更新：2026-08-10  
> 用途：先定义成品价值与验收标准，再约束后续组件实现，避免局部设计偏离项目主线

## 1. 项目定位

**DAO Agent Harness** 是一套面向开发者的轻量智能体运行框架。它不以预装大量应用功能为目标，
而是提供让 Agent 能够可靠运行、持久恢复、治理上下文、扩展能力和安全协作的运行内核。

一句话介绍：

> 参考 nanobot 的 Agent Loop 工程经验与 pi agent-core 的组件边界，自主设计一套强类型、可持久化、
> 可恢复、可扩展的 Agent Harness。

核心价值按优先级排列：

1. **可靠运行**：输入不丢失、历史可重放、失败不伪造成功、恢复边界明确；
2. **上下文治理**：在 token 预算内组合对话尾部、摘要、长期记忆、Skill 和外部知识；
3. **能力扩展**：本地工具、MCP、RAG 和 Skill 使用统一边界接入，不侵入 Runner；
4. **执行控制**：工具权限、超时、审批、沙盒、预算、取消与副作用状态可治理；
5. **协作与观测**：子 Agent 具有隔离上下文和任务契约，所有关键运行过程可追踪和评测。

## 2. 成品体验

DAO Agent Harness 的目标使用体验是：

1. 应用通过 Python SDK、CLI 或 API 提交用户输入；
2. Runtime 先持久化输入，再构建上下文并启动 Agent Loop；
3. Runner 以流式方式调用 Provider，根据 Tool Registry 执行本地或 MCP 工具；
4. 长对话自动生成分支相关 ContextSummary，模型只接收摘要与未覆盖消息尾部；
5. Memory 保存跨会话稳定事实和经验，RAG 提供可引用的外部知识；
6. Skill 描述可复用工作流程、上下文资源和工具依赖；
7. 复杂任务可以委派给具有独立 Session、权限和预算的子 Agent；
8. 暂停、失败或进程重启后，Runtime 从安全 ContextCheckpoint 恢复；
9. 开发者可以查看模型请求、token、工具耗时、失败原因和任务树。

产品主链路：

```text
CLI / SDK / API
      │
      ▼
AgentRuntime
├─ PendingInput / Session Event Log
├─ ContextCheckpoint / Recovery
├─ SessionContextResolver
├─ ContextBuilder
│  ├─ ContextSummary
│  ├─ Long-term Memory
│  ├─ RAG Context
│  └─ Active Skills
├─ AgentRunner
│  ├─ Provider
│  └─ ToolRegistry
│     ├─ Local Tools
│     └─ MCP Adapter
└─ Multi-Agent Orchestrator
   ├─ Child Sessions
   ├─ Shared Artifacts
   └─ Structured Results
```

## 3. 关键概念边界

| 组件 | 保存或处理的内容 | 主要消费者 | 不负责什么 |
|---|---|---|---|
| ContextSummary | 当前 Session 某个分支的压缩历史 | ContextBuilder | 跨会话长期知识、任务恢复 |
| Memory | 用户事实、偏好、稳定决策和可复用经验 | ContextBuilder / Memory Retriever | 原始对话备份、外部知识库 |
| RAG | 从外部文档或知识库检索出的证据 | ContextBuilder / Agent | 用户长期偏好、执行恢复 |
| Skill | 工作流程、说明、资源和工具依赖 | ContextBuilder / Skill Runtime | 直接执行工具、保存会话 |
| ToolRegistry | 工具发现、Schema、校验、执行策略和结果 | Runner | 模型对话、长期记忆 |
| MCP Adapter | MCP Tool、Resource、Prompt 到内部协议的映射 | ToolRegistry / ContextBuilder | 重写 Runner 循环 |
| ContextCheckpoint | Runner 阶段、游标和恢复所需技术状态 | AgentRuntime | 模型历史摘要 |
| Multi-Agent | 有边界的子任务委派、预算、取消和结果回传 | Orchestrator / Parent Agent | 任意共享可变 Session 状态 |

当前路线中的“外部知识检索”统一称为 **RAG**。如果未来引入名为 Rack 的具体协议或组件，应作为
独立集成重新定义，不能与 RAG 混用。

## 4. 架构原则

### 4.1 事实与视图分离

- Conversation Entry Tree 和 Session Event Log 是持久化事实；
- ContextSummary、Memory 检索结果和 RAG 结果是模型上下文组成部分；
- ContextGovernor 只生成单次 Provider 请求视图，不改写正式历史；
- ContextCheckpoint 保存执行恢复状态，不冒充对话消息。

### 4.2 Runner 保持小而稳定

Runner 只负责：

```text
Provider Request
→ Model Response
→ Tool Calls
→ Tool Results
→ Continue / Stop
```

Session、Memory、RAG、MCP 生命周期、多 Agent 调度和应用 Transport 不进入 Runner 核心。

### 4.3 扩展能力必须声明边界

每个 Tool、Skill、MCP Server 和子 Agent 至少声明：

- 输入输出 Schema；
- 所需权限；
- 并行或串行属性；
- token、轮次或时间预算；
- 可取消性；
- 是否可能产生外部副作用；
- 失败后的状态是否已知。

### 4.4 生成内容必须经过领域校验

模型生成的 ContextSummary、Memory 记录、Skill 草案和 AgentTask 结果不能直接成为可信系统状态，
必须经过强类型解析、Schema 校验、大小限制和明确的失败语义。

## 5. 当前真实完成度

### 5.1 已实现

- Provider-neutral completion 与统一流事件；
- OpenAI-compatible Provider 和 SSE 工具调用聚合；
- 强类型 AgentMessage、ToolCall 和 ToolResult；
- Tool Registry、JSON Schema 参数校验、并行安全标签和批次执行；
- `LOAD → PREPARE → RUN → SAVE → RESPOND` Runtime；
- PendingInput 预保存、编辑、失败保留和队列顺序；
- Message Entry Tree、Active Leaf、分支继承和原子 TurnCommitted；
- JSON Snapshot 与 append-only JSONL Session Event replay；
- 结构化 ContextSummary、严格 JSON 解析和一次修复；
- 分支感知 SessionContextResolver；
- nanobot 式 token budget、50% 压缩目标、User Turn 边界和最多 5 轮 Consolidator；
- Provider counter 到可选 tiktoken 的事前 token estimator chain；
- Runner ContextGovernor 的 NORMAL / PRESSURE / EMERGENCY 三级上下文兜底；
- CLI Adapter、流式渲染、运行中追加、`/pause`、`/edit`、`/resume` 与 JSONL 多轮恢复；
- Tool timeout、内容寻址 ArtifactStore、Session Codec v6 和 `read_artifact` 分页回读；
- 三节点 ContextCheckpoint、独立原子 Store、保守工具恢复和 SAVE 失败续提；
- PendingInput 双安全点消息插入、五条总配额、多输入 Checkpoint 与新 Runner 交接；
- SkillCatalog、`activate_skill`、`read_skill_resource`、Skill 指令保护与压缩后恢复；
- Runtime Status 强类型快照、user-role 模型投影与 Session 持久化；
- Workspace 长期记忆、强类型 Memory Inbox、两阶段 Dream、失败游标和 ContextBuilder 注入；
- 276 个自动化测试与 Ruff 静态检查。

### 5.2 尚未实现

- 跨进程暂停快照与任意阶段恢复；
- RAG、MCP，以及 Skill Package、安装、依赖和 Authoring；
- 多 Agent 委派；
- 工具 approval、sandbox 和副作用恢复；
- 运行 trace、评测体系和生产级数据库 Store。

## 6. Roadmap 与验收标准

总体验收矩阵：

| 里程碑 | 要证明的项目能力 | 当前状态 | 简历条目启用条件 |
|---|---|---|---|
| M0 Harness Core | Loop 之外具备可靠消息、Session、工具、上下文与保存边界 | 已完成 | 当前真实版本可使用 |
| M1 Single-Agent Runtime | 可供 CLI 调用并治理上下文与工具；安全点恢复后续增强 | 进行中（CLI v1 已验收） | Runtime、Checkpoint、ContextGovernor 和 trace 验收全部通过 |
| M2 Memory & RAG | 同时具备跨会话连续性和有引用的外部知识 | 进行中（Memory v0.1 已实现） | Memory 来源/删除与 RAG 引用/质量测试通过 |
| M3 MCP & Skills | 能力可在不修改 Runner 的情况下接入、编写和受控激活 | 待实现 | MCP 离线降级、Skill 依赖/验证/审批链通过 |
| M4 Multi-Agent | 主 Agent 能以权限和预算受控的契约委派子任务 | 待实现 | 隔离、预算、取消、结果协议和任务树 trace 通过 |
| M5 Production | 具备数据库、迁移、安全、Eval、SDK/API 和发布能力 | 待实现 | 真实 benchmark 与发布检查通过 |

### M0：可靠的 Harness Core——已完成

目标：证明 DAO Agent Harness 不只是一个 while-loop，而是具有消息、Session、工具、上下文和保存
边界的最小运行内核。

验收：

- [x] Provider、Runner、ToolRegistry、ContextBuilder、Runtime 可独立测试；
- [x] 用户输入在模型调用前持久化；
- [x] 成功 Turn 原子提交，失败保留 PendingInput；
- [x] 对话可分支且完整事实树不被摘要覆盖；
- [x] 长上下文可以生成耐久的结构化 ContextSummary；
- [x] 完整自动化测试通过。

### M1：应用可用的单 Agent Runtime——进行中

目标：让真实应用能够稳定使用 DAO Agent Harness 完成多轮工具任务。

计划：

- CLI 或最小 Transport Adapter 接入 AgentRuntime；
- Runner ContextGovernor：工具链修复、micro-compaction、tool result budget、emergency snip；
- 工具 timeout、approval、sandbox 和大结果 ArtifactStore；
- [x] `awaiting_tools`、`tools_completed`、`final_response` 三个 ContextCheckpoint；
- [x] PendingInput 双安全点注入、五条总配额和剩余输入新 Runner 交接；
- [x] 主动暂停、局部消息修订和未知副作用提示；
- 跨进程暂停快照与更完整的副作用保护；
- 运行 Timeline、结构化日志、token 与阶段耗时。

三节点 ContextCheckpoint 与消息插入已完成；详细设计分别见
[ContextCheckpoint 设计](checkpoint-design.md)和[消息插入设计](message-injection-design.md)。
主动暂停和局部修订也已完成，见[主动暂停与消息修订设计](pause-revision-design.md)。

工具结果的通用语义边界已经实现：ToolOutput 分离模型视图、完整 Artifact、显式错误和 Harness
metadata，Registry 对未知工具保留通用外置与预览 fallback。后续不需要先建立统一结果适配器；
文件、Shell、搜索等具体工具可以按自身语义逐个提供摘要、结构化视图或完整 Artifact。

验收：

- [x] CLI 两轮对话、工具调用和流式回答经过 Runtime/Session 主链路；
- [x] Provider 失败后重新运行无需用户重发输入；
- [x] 三个安全节点均有恢复测试；
- [x] 未知状态工具不会被自动重复执行；
- [x] 被提交的追加输入至少进入过一次 Provider 请求；
- [x] 超窗和超大 ToolResult 有可预测降级行为；
- [ ] 每轮执行可从 trace 定位模型、工具、保存和失败阶段。

### M2：Memory 与 RAG

目标：同时解决跨会话连续性和外部知识检索，但保持两者存储与可信度边界独立。

计划：

- 强类型 MemoryInboxEntry、MemoryPlan 与 DreamRunRecord；
- nanobot 式两阶段 Dream，以及人可读、可手工修订的 Workspace `MEMORY.md`；
- 用户事实、偏好、决定和成功经验的提取、去重、修正与删除；
- Memory Inbox 游标、失败重试、来源追踪和后台调度；
- Retriever Protocol、Document、Chunk、Citation；
- 文档摄取、索引、召回和结果重排；
- ContextBuilder 的 Summary、Memory、RAG 预算分配与去重。

验收：

- [x] 新 Session 能恢复明确保存的用户偏好；
- [x] Memory 变更可通过 DreamRunRecord 追踪来源并由后续 Dream 删除；
- [ ] RAG 回答能返回文档引用；
- [ ] 外部文档中的指令不会提升为 System Prompt；
- [ ] Summary、Memory 和 RAG 不保存同一类信息的无界副本；
- [ ] 有检索质量与 token 开销基准测试。

Memory v0.1 的具体边界、输出协议和实现顺序见
[长期记忆与 Dream 设计](memory-system-design.md)。第一版不引入 MemoryRecord 数据库、向量检索、
自动 Skill 生成或对 USER.md/SOUL.md 的自动修改。

### M3：MCP 与 Skill 生态

目标：不修改 Runner 即可扩展外部工具、资源、Prompt 和可复用工作流程。

计划：

- MCP Client 生命周期、连接状态和能力发现；
- MCP Tool 映射到 ToolRegistry；
- MCP Resource 和 Prompt 映射到 ContextBuilder；
- Skill Package、元数据、资源、依赖和激活条件；
- Skill Loader、冲突处理、版本和缓存；
- Skill Authoring：Draft → Validate → Test → Review/Approve → Activate。

验收：

- [ ] MCP Server 离线不会破坏本地工具注册；
- [ ] MCP 工具错误按普通 ToolResult 回流模型；
- [ ] Skill 可以声明工具和 MCP 依赖；
- [ ] Skill 草案未经验证和审批不能覆盖已激活版本；
- [ ] 新增 Skill 不需要修改 Runner；
- [ ] 至少提供一个端到端 MCP 示例和两个可复用 Skill。

### M4：有边界的多智能体协作

目标：支持主 Agent 委派子任务，而不是建设通用 DAG 工作流平台。

计划：

- AgentTask、AgentResult 和 AgentProgress；
- Child Session 与可控上下文继承；
- 工具白名单、token/轮次/时间预算；
- 父子取消传播和失败回收；
- Shared Artifact 引用；
- 并行子任务和结构化结果汇总。

验收：

- [ ] 子 Agent 不能访问未授权工具和 Session 分支；
- [ ] 子任务达到预算后产生闭合结果；
- [ ] 父任务取消可以终止子任务；
- [ ] 子 Agent 通过结果协议回传，不直接修改父 Session 历史；
- [ ] 并行结果不依赖完成顺序；
- [ ] trace 能展示完整父子任务树。

### M5：质量、安全与产品化

计划：

- SQLite 或数据库 SessionStore；
- snapshot、日志压缩、migration 和多进程 lease；
- SDK、HTTP API、示例应用与配置体系；
- secrets、PII、Prompt Injection 和工具权限治理；
- Eval Dataset、回归测试、token/时延/成功率指标；
- 文档、版本策略和发布流程。

验收指标必须通过 benchmark 或真实测试产生，不能为简历预设虚假数字。

## 7. 简历项目经历

### 7.1 当前即可使用的真实版本

**DAO Agent Harness｜轻量、可持久化的智能体运行框架**  
`Python` · `asyncio` · `Typed Domain Model` · `JSONL Event Sourcing` · `OpenAI-compatible API`

参考 nanobot 的 Agent Loop 工程实践和 pi agent-core 的 Provider、消息与工具边界，自主设计轻量
Agent Harness，将模型调用、工具执行、上下文构建、会话持久化和外部编排拆分为可测试组件。

- 设计 `LOAD → PREPARE → RUN → SAVE → RESPOND` 分阶段 Runtime，通过 PendingInput 预保存、
  per-Session execution lock 和原子 TurnCommitted，保证模型失败后用户输入与正式历史不丢失；
- 使用强类型 AgentMessage、Message Entry Tree 和 append-only JSONL Event Log 表达对话、编辑与
  分支，支持从历史节点创建继承上下文的新分支；
- 实现结构化 ContextSummary 和 token-budget Consolidator，通过严格 JSON 校验、修复重试和分支
  Resolver，在保留完整事实树的同时缩减模型请求历史；
- 构建 Provider-neutral 流式事件和 Tool Registry，统一处理 SSE 工具参数聚合、Schema 校验、
  并行安全批次、错误结果回流和 `tool_call_id` 关联；
- 建立 276 个自动化测试，覆盖 Provider、Runner、工具并发、Session replay、上下文压缩、
  Artifact 外置回读、ContextCheckpoint 恢复、消息插入和失败一致性。

### 7.2 成品版目标模板

以下条目只有通过相应 Roadmap 验收后才能写入正式简历：

- 构建统一上下文装配层，按 token 预算组合近期对话、ContextSummary、跨会话 Memory、RAG 检索结果
  和 Active Skills，并对来源、优先级、引用和 Prompt Injection 边界进行治理；
- 通过 MCP Adapter 将远程 Tool、Resource 和 Prompt 映射到 Harness 内部协议，使扩展能力无需侵入
  Runner 核心；
- 设计 Skill Package 与受控 Authoring 流程，支持依赖声明、静态校验、测试、版本化、审批和激活；
- 在三个安全节点保存 ContextCheckpoint，实现暂停、失败和进程重启后的任务恢复，并避免重复执行
  状态未知的外部副作用；
- 基于 Child Session、任务契约、权限白名单和执行预算实现多 Agent 委派，通过 Shared Artifact 和
  结构化结果隔离父子任务状态；
- 建立运行 trace 与 Eval，持续衡量 token、时延、工具成功率、恢复率和任务完成质量。

### 7.3 简历数据规则

- “已实现”“支持”“降低”“提升”等表述必须有代码和测试支撑；
- 自动化测试数量可以使用真实测试结果；
- token 节省比例、恢复成功率、工具接入时间和任务成功率必须来自固定 benchmark；
- 规划中的能力保留在本文件，不提前写成已经完成的简历成果；
- 每完成一个里程碑，先更新验收矩阵，再更新成品版简历条目。

## 8. 明确非目标

近期不把以下能力作为 DAO Agent Harness 的主线：

- 通用 DAG 或低代码工作流平台；
- 插件市场和商业分发平台；
- 大量聊天 Channel；
- 完整 WebUI；
- 任意事件点的时间旅行；
- exactly-once 外部副作用承诺；
- 复杂的多 Agent 组织社会或无限自治；
- 与 Harness 可靠运行无直接关系的应用功能集合。

这些能力只有在具体应用或验收目标提出真实需求后，才进入 Roadmap。

## 9. 文档维护约定

本文件是 DAO Agent Harness 的方向约束：

1. 新组件必须说明它服务于哪个成品能力；
2. 新功能如果不能对应某项验收或简历价值，应先质疑其必要性；
3. 深入实现某个局部前，要确认它没有阻塞更高优先级里程碑；
4. 代码完成不等于能力完成，必须同时具备失败测试、集成测试和文档；
5. Roadmap 可以调整，但概念边界和真实简历原则不能被绕过。
