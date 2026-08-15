# 第一阶段验收：Single-Agent CLI Harness

> 状态：Accepted  
> 日期：2026-08-10  
> 第一阶段测试基线：156 passed；当前基线：250 passed，Ruff clean

## 1. 验收结论

DAO Agent Harness 第一阶段已经形成可运行、可持久化、可测试的单 Agent CLI 闭环。它不再只是
最小 Agent Loop，而是具备明确消息边界、失败保存语义、上下文治理、工具执行策略和大型结果外置
能力的 Harness Core。

本阶段可以正式结束。运行 Timeline、approval 和 sandbox 是后续可靠性与安全增强，不作为第一
阶段完成的阻塞条件；ContextCheckpoint 已被选为紧接第一阶段的下一项实现。

## 2. 已闭合的主链路

```text
CLI RuntimeRequest
  → PendingInput 预保存
  → Session 分支上下文解析
  → ContextSummary + ContextBuilder
  → ContextGovernor
  → Provider 流式生成
  → ToolRegistry 校验、timeout 与批次执行
  → 大结果 ArtifactStore 外置
  → read_artifact 分页回读
  → TurnCommitted 原子保存
  → CLI 响应
```

失败时，未完成 Turn 不进入正式 Message Entry Tree，PendingInput 保留并可通过 `/retry` 再次执行。

## 3. 验收矩阵

| 能力 | 状态 | 证据 |
|---|---|---|
| Provider-neutral 非流式与流式调用 | 完成 | OpenAI-compatible Provider、SSE 聚合测试 |
| 模型—工具循环 | 完成 | 多轮、错误、并行与 sequential 屏障测试 |
| 输入不丢失 | 完成 | PendingInput 预保存、失败保留、CLI `/retry` 测试 |
| Session 强类型持久化 | 完成 | Snapshot、JSONL Event replay、尾记录恢复测试 |
| 对话分支 | 完成 | Entry Tree、Active Leaf、checkout 与继承测试 |
| 上下文压缩 | 完成 | 结构化 ContextSummary、修复、分支 Resolver 测试 |
| 临时上下文治理 | 完成 | 工具链修复、micro-compaction、裁剪与 emergency snip 测试 |
| 工具执行策略 | 完成 | schema 校验、工具级 timeout、错误结果回流测试 |
| 大结果治理 | 完成 | Artifact 去重、完整性校验、Registry 外置、Codec v6 测试 |
| Skill 基础能力 | 完成 | Catalog、渐进加载、资源安全、metadata 与压缩后恢复测试 |
| CLI Artifact 回读 | 完成 | LocalArtifactStore、`read_artifact` 端到端测试 |
| 主动暂停与局部修订 | 完成 | ActiveRun、paused latch、消息截断与 CLI `/pause` 测试 |
| Runtime Status | 完成 | 强类型快照、Runner 注入、状态持久化和压缩过滤测试 |
| 静态与回归质量 | 完成 | 第一阶段 156 passed；当前 250 passed，Ruff clean |

## 4. 已知边界

- 当前只有 OpenAI-compatible Provider 实现；
- CLI 是唯一正式 Adapter，尚无 HTTP、GUI 或通用 Transport；
- AgentMessage 仍以纯文本为主，不支持多模态 block；
- JSON Schema 校验是满足当前工具协议的子集；
- SessionStore 以单进程语义为边界，没有数据库 lease；
- 暂停工作前缀只在当前进程保留，尚无跨进程暂停快照、运行 Timeline 和外部副作用状态机；
- 没有 Memory、RAG、MCP、Skill Package/Authoring 和多 Agent。

这些限制均有明确组件边界，不破坏第一阶段已有语义。

## 5. 暂缓项

以下内容继续保留设计，不立即实现：

- 持久化 Run Timeline 和 trace；
- approval、sandbox 和未知外部副作用恢复；
- 多进程 Session 协调。

当前文件和 Shell 工具只是能力骨架，尚不面向真实生产工作负载；approval、sandbox 和复杂外部
副作用恢复等机制等到工具进入实际使用阶段再设计。

## 6. 阶段后续进展

`awaiting_tools`、`tools_completed`、`final_response` 三节点 ContextCheckpoint 已实现并完成
恢复测试，设计见 [ContextCheckpoint 设计](checkpoint-design.md)。双安全点消息插入也已完成，见
[消息插入设计](message-injection-design.md)；运行 Timeline、MCP、Memory 与 Skill 留待后续排序。
主动暂停、追加消息局部修订与 CLI 交互也已完成，见
[主动暂停与消息修订设计](pause-revision-design.md)。
