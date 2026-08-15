# Runtime Status 设计

> 状态：Implemented v0.1  
> 最后更新：2026-08-14  
> 范围：通用 Harness 的模型可见运行状态框架，不包含具体业务状态

## 1. 定位

Runtime Status 是 Runner 在每次真实模型决策前捕获的一份运行快照。它解决的不是 UI 状态展示，
也不是通用任务管理，而是向模型补充三类仅靠对话记录无法可靠获知的信息：

- 当前时间与时区；
- 当前执行轨迹中的重复工具异常；
- ContextGovernor 为满足窗口预算而隐藏过的上下文。

状态使用强类型 `RuntimeStatusSnapshot` 表达，渲染为 `RuntimeStatusMessage` 后进入 working messages，
并在成功 SAVE 时随本轮完整消息段持久化。Provider 仍只接收标准 `user`、`assistant`、`tool`
角色；`RuntimeStatusMessage` 在 Provider 边界映射为 `role=user`。

```text
typed working messages
        ↓
ContextGovernor.prepare()
        ↓  单次请求的治理报告
RuntimeStatusBuilder
        ↓
RuntimeStatusMessage（追加到 working messages）
        ↓
ContextBuilder → role=user → Provider
```

## 2. 为什么保持独立消息

状态不拼进原始 `UserMessage`，原因如下：

- 用户消息身份、PendingInput 消费和编辑规则保持不变；
- 暂停修订可以按消息边界删除状态及其后的模型轨迹；
- Session 能审计模型当时实际看到了什么；
- Entry Tree 分支自然继承分叉点以前的状态；
- ContextGovernor 可以在压力下只删除过期状态，不改写用户文本；
- 未来 GUI 可以根据 `display=False` 隐藏状态，不影响模型协议。

内部类型独立不等于新增 Provider 角色。`ContextBuilder` 将它投影为标准 `user` 消息；若它紧邻
普通 UserMessage，两者只在临时 Provider 视图中合并，正式 Session 仍保留两个强类型节点。

## 3. 强类型协议

```python
RuntimeStatusSnapshot(
    schema_version=1,
    environment=RuntimeEnvironmentStatus(...),
    tool_anomalies=(ToolAnomalyStatus(...), ...),
    context_visibility=ContextVisibilityStatus(...) | None,
)

RuntimeStatusMessage(
    snapshot=...,
    content="<dao_runtime_status ...>",
    render_profile="dao-default-v1",
    display=False,
)
```

`snapshot` 供 Harness 做校验、治理和未来 UI 投影；`content` 保存当时实际发送给模型的稳定文本。
历史消息不会因为未来渲染器升级而被重新渲染。`render_profile` 标识渲染协议，`display` 只表达
默认 Adapter 是否应把它当普通对话展示。

Session Codec v6 完整编码消息、快照、渲染 profile 和 display 标记，并继续读取 v1-v5 Session。
JSONL Event Store 不新增专用事件；状态和本轮其他消息一起由 `TurnCommitted` 原子提交。

## 4. 默认内容

### 4.1 Environment

每份状态都包含：

- timezone-aware ISO 8601 当前时间；
- 当前时间函数给出的时区名称。

模型不能从静态 system prompt 得知当前时间，因此它属于动态状态。workspace、操作系统等相对
稳定信息更适合未来放入静态 Environment system section，当前版本不重复注入。

### 4.2 Tool anomalies

只有异常才出现，不提供普通工具调用计数：

- `repeated_identical_call`：同一工具与 canonical JSON 参数在当前执行轨迹中至少出现两次；
- `repeated_failure`：同一工具在当前执行轨迹中至少失败两次。

状态只告诉模型工具名、异常类型和次数，不复制参数或错误正文；具体调用和结果仍以正式
Assistant/ToolResult 消息为准。阈值可在 `RuntimeStatusBuilder` 构造时调整。

### 4.3 Context visibility

正常上下文不显示此段。只有模型本次看到的内容与完整 working messages 不同且存在信息损失时，
才提供：

- `pressure`：旧工具结果被 micro-compaction，或旧 Runtime Status 被丢弃；
- `emergency`：历史消息或当前执行轨迹被 emergency snip；
- 各类被压缩、隐藏消息的数量。

单个超长工具结果的确定性首尾裁剪没有单独进入状态栏；裁剪标记已经直接存在于模型可见的
ToolResult 中，不需要重复描述。

## 5. 默认渲染协议

```xml
<dao_runtime_status version="1" source="harness" authority="metadata">
  <environment>
    <current_time>2026-08-14T16:30:00+08:00</current_time>
    <timezone>Asia/Shanghai</timezone>
  </environment>
  <tool_anomalies>
    <anomaly kind="repeated_failure" tool="search" occurrences="2" />
  </tool_anomalies>
  <context_visibility mode="pressure">
    ...
  </context_visibility>
</dao_runtime_status>
```

Tool Contract 明确规定：该标签是 Harness metadata，不是用户撰写的指令；多个状态同时可见时，
最新状态对时间、工具异常和上下文可见性具有权威性。XML 值经过转义，渲染顺序固定。

## 6. Runner 生命周期

每个模型 iteration 的顺序是：

1. 从 `context_prefix_messages + working tail` 构建可见强类型消息；
2. ContextGovernor 修复工具协议并按预算治理单次模型视图；
3. RuntimeStatusBuilder 读取当前 Turn 的完整 working trace 和本次治理报告；
4. 将新 RuntimeStatusMessage 追加到 working messages；
5. 将治理视图加上最新状态投影为 Provider 消息；
6. 发起一次真实 Provider 请求。

Provider 内部 HTTP retry 不会生成额外状态；只有 Runner 再次请求模型才产生下一份快照。
Checkpoint 分别保存在 Assistant tool calls、ToolResult 完成和最终 Assistant 后，因此自然携带最近
状态，不新增第四个 checkpoint 节点。

执行失败或取消时 Runtime 不提交本轮 working tail，状态与未完成生成一起丢弃，PendingInput 仍
保留。成功或 `limit_reached` 时，状态随完整 Turn 提交。主动暂停后的局部修订会删除目标用户消息
之后的状态、Assistant 和 ToolResult，再从修订点重新运行。

## 7. ContextGovernor 与 Consolidator

正常预算下历史状态保持 append-only，使模型请求前缀稳定并保留可审计语义。请求超预算后，
ContextGovernor 在压缩工具结果和 emergency snip 之前先删除旧 RuntimeStatusMessage；本轮最新状态
在治理完成后才生成，因此始终保留。

Runtime Status 是短期运行事实，不进入长期 `ContextSummary`。ContextSummaryGenerator 在生成
摘要前过滤状态 Entry，但 Summary 的覆盖边界仍可跨过这些 Entry。这样 Entry Tree 保留真实历史，
摘要只保存用户目标、事实、决策、工作进度和问题，不会固化过期时间或某次临时裁剪状态。

## 8. 明确不包含的字段

v0.1 不包含：

- 剩余模型 iteration；
- 已吸收或仍待处理的用户追加消息数量；
- UI loading、streaming、spinner 等展示状态；
- Token 调试明细和内部对象 ID；
- Goal、TODO、Memory、MCP、Subagent、Approval 或业务流程状态。

前两项由代码硬限制且模型无需知道；频繁达到 max_turns 应被视为 Harness/工具设计问题，而不是
靠状态提示补救。后续业务只有满足“模型无法直接观察、推断成本高且会影响下一步决策”时，才应
扩展快照或通过独立领域组件提供状态。

## 9. 扩展边界

当前没有通用 StatusManager、插件系统或业务字段注册表。未来适配具体业务时可以：

- 替换 `RuntimeStatusRenderer`，适配特定模型的稳定标签；
- 在上层生成独立的任务/审批/子代理状态模块；
- 将同一 Snapshot 投影为 GUI 状态卡，而不从对话文本反向解析；
- 为业务状态建立权威 Store，再由状态栏只投影模型真正需要的部分。

事实存储、模型视图和 UI 展示保持三层分离。Runtime Status 是模型视图的通用基础，不承担整个
应用的状态管理职责。

## 10. 不变量

1. 每次真实模型决策最多追加一条新状态；
2. 状态不伪装成用户原始输入，PendingInput 身份不变；
3. Provider 只看到标准角色；
4. 保存的是实际渲染文本，不对历史重新渲染；
5. 最新状态不会被本次 ContextGovernor 删除；
6. ContextSummary 不吸收过期运行状态；
7. 状态扩展不能绕过 system、用户授权或工具安全边界；
8. 默认状态不包含业务字段和代码已强制执行的内部限额。
