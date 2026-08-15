# 工具注册、校验与执行设计

> 状态：Implemented v0.5  
> 最后更新：2026-08-14  
> 范围：AgentTool、ToolRegistry 与 AgentRunner 之间的工具调用边界

## 1. 设计结论

当前 Harness 不引入独立 `ToolRuntime`。工具子系统采用三层分工：

```text
AgentTool
  └─ 声明 schema、实现、execution_mode 和可选 timeout

ToolRegistry
  └─ 注册、查找、参数转换、校验、单次执行、超时和结果归一化

AgentRunner
  └─ 批次划分、并发调度、ToolResultMessage 转换和消息追加
```

这保留了 nanobot 的核心思想：并发能力属于工具自身；Registry 在执行前准备调用；Runner
控制工具批次并把普通错误作为观察结果交回模型。DAO v0.2 仍不引入独立 ToolRuntime，而是在
Registry 内增加统一的单调用执行协议；approval、sandbox 或远程执行显著复杂化时，再考虑从
Registry 内部提取执行器。

## 2. nanobot 参考与精简

nanobot 的 Tool 使用三个属性描述并发语义：

```text
read_only
concurrency_safe
exclusive
```

其中 `concurrency_safe` 默认由 `read_only and not exclusive` 推导。它的 ToolRegistry 通过
`prepare_call()` 完成工具解析、schema 驱动的参数转换和校验；Runner 再把连续的并发安全
调用组成批次，将其他调用作为独立屏障。nanobot 当前没有统一的 Runner 级工具 timeout：Shell、
MCP、Web 等组件分别处理自己的超时。DAO 保留其批次和错误回流思想，但把通用 deadline 统一到
ToolRegistry 的单调用执行入口，避免每种工具重复实现 Harness 级超时语义。

本项目把三个属性精简为一个明确标签：

```python
ToolExecutionMode = Literal["parallel_safe", "sequential"]
```

- `parallel_safe`：允许与相邻的同类调用同时执行；
- `sequential`：必须单独执行，并阻断前后并行批次合并。

`sequential` 只约束同一模型响应中的工具批次，不表示全局分布式锁。

## 3. AgentTool

工具最小协议为：

```python
class AgentTool(Protocol):
    name: str
    description: str
    parameters: Mapping[str, Any]
    execution_mode: ToolExecutionMode
    timeout_s: float | None

    async def execute(self, arguments: Mapping[str, Any]) -> Any: ...
```

工具负责：

- 提供稳定名称和模型可见说明；
- 声明参数 schema；
- 声明并发标签；
- 可选声明自己的正数 timeout；未声明或为 None 时继承 Registry 默认值；
- 实现具体能力；
- 在已知的业务失败场景下，返回 `ToolOutput(is_error=True)`。

工具不负责查找其他工具、划分批次、构造 ToolResultMessage 或决定 Agent Loop 是否继续。模型
参数不能覆盖或延长 Harness timeout；需要较长运行时间的工具由开发者显式配置，未来的长任务
则应返回任务句柄，而不是无限占用一次工具调用。

## 4. ToolRegistry

ToolRegistry 是工具定义、调用准备和单次执行的唯一入口。

### 4.1 ToolExecutionPolicy

```python
@dataclass(frozen=True, slots=True)
class ToolExecutionPolicy:
    default_timeout_s: float | None = 60.0
```

- 正数表示未单独配置工具的默认 deadline；
- None 表示 Registry 不施加通用 timeout；
- 0 或负数属于配置错误；
- 工具自己的 `timeout_s` 优先于 Registry 默认值；
- 第一版不允许模型参数改变 deadline，也不增加 batch timeout。

每工具独立 deadline 可以让同一并行批次中的其他工具继续完成；batch timeout 会把一个慢工具的
策略错误扩大到整个批次，因此暂不采用。

### 4.2 注册

`register()` 校验：

- 工具名称非空且不重复；
- `execution_mode` 只能是 `parallel_safe` 或 `sequential`；
- timeout 如果存在，必须是正数；
- parameters 是顶层 `type=object` 的 mapping。

`definitions()` 返回独立 schema 副本，调用方不能通过修改 Provider 请求反向改变已注册工具。

### 4.3 prepare_call

```python
prepare_call(
    name,
    arguments,
) -> tuple[AgentTool | None, dict[str, Any], str | None]
```

顺序固定为：

```text
按名称解析工具
→ 根据参数 schema 做安全转换
→ 校验转换后的参数
→ 返回 tool、prepared arguments、error
```

当前实现支持工具参数所需的常用 JSON Schema 子集：

- `object`、`array`、`string`、`integer`、`number`、`boolean`；
- nullable type；
- `required`、`additionalProperties=false`；
- `enum`；
- `minimum`、`maximum`；
- `minLength`、`maxLength`；
- `minItems`、`maxItems` 和嵌套 `items`。

安全转换包括字符串到 integer、number、boolean，以及按 schema 递归转换 object 和 array。
它不是完整 JSON Schema Draft 实现；未来需要正则、联合 schema 或引用解析时再评估专用库。

未知工具和参数错误不抛出普通执行异常，而是返回可转换为 ToolResult 的错误文本。

### 4.4 ToolExecutionResult

Registry 的公开执行结果为：

```python
ToolExecutionStatus = Literal["completed", "failed", "timed_out"]
ToolExecutionErrorCode = Literal[
    "not_found",
    "invalid_arguments",
    "exception",
    "reported_error",
    "timeout",
    "artifact_store",
]

@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    tool_call_id: str
    tool_name: str
    content: str
    status: ToolExecutionStatus
    error_code: ToolExecutionErrorCode | None = None
    artifact_refs: tuple[ArtifactRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.status != "completed"
```

`error_code` 用于稳定测试和未来 Timeline，不直接决定 Agent Loop 是否结束，也不持久化独立诊断
事件。正常的非字符串结果由 Registry 使用 UTF-8 JSON 归一化；不能编码时退回 `str()`。
当成功结果超过配置阈值时，Registry 通过 ArtifactStore 外置原文，并在结果中保留头尾预览和
强类型引用；详细协议见 [ArtifactStore 设计](artifact-store-design.md)。

工具需要控制模型视图、完整结果、错误语义或 Harness-only metadata 时，返回：

```python
@dataclass(frozen=True, slots=True)
class ToolOutput:
    content: str
    artifact_content: str | None = None
    is_error: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    allow_externalization: bool = True
```

- `content` 是工具主动设计的模型视图；
- `artifact_content` 是可选完整结果，存在时无论长度都必须写入 ArtifactStore；
- `is_error` 显式表示工具主动报告失败；
- `metadata` 只供 Harness、Session、ContextGovernor 或未来 UI 使用；
- `allow_externalization=False` 用于 Skill 指令和 Artifact 分页结果等不得再次外置的内容。

`artifact_content` 可以与 `is_error=True` 同时使用，以保存命令、测试或编译失败的完整日志；
但不能与 `allow_externalization=False` 同时使用。未返回
ToolOutput 的工具仍可返回字符串或 JSON-compatible 值，Registry 使用原有通用归一化和大型结果
外置 fallback。

### 4.5 execute_call

```python
async def execute_call(self, call: ToolCall) -> ToolExecutionResult: ...
```

执行顺序固定为：

```text
prepare_call
→ 若失败，返回 failed ToolExecutionResult
→ 解析 tool timeout 或 Registry 默认 timeout
→ 在 deadline 内 await tool.execute(prepared_arguments)
→ 归一化结果或异常
→ 处理显式错误、工具模型视图和完整 Artifact
→ 对未适配的大型文本结果执行通用 Artifact 外置
→ 返回 ToolExecutionResult
```

ToolRegistry 不构造 ToolResultMessage。Runner 收到 ToolExecutionResult 后，使用原 call ID 和
`is_error`、`artifact_refs`、`metadata` 构造领域消息。

SkillCatalog 不属于 ToolRegistry。`activate_skill` 和 `read_skill_resource` 注册为 AgentTool，
具体发现、读取与安全路径解析委托给内部 SkillCatalog。完整协议见
[Skill 系统设计](skill-system-design.md)。

## 5. Runner 批次划分

Runner 按模型给出的调用顺序扫描：

```text
parallel A ─┐
parallel B ─┘ batch 1：并行

sequential C   batch 2：单独执行

parallel D ─┐
parallel E ─┘ batch 3：并行
```

规则：

1. 连续 `parallel_safe` 调用进入同一批次；
2. 遇到 `sequential`、未知工具或不可识别工具时，先结束当前并行批次；
3. 非并行调用形成单元素批次；
4. 批次严格依次执行，后一个批次不能越过前一个批次；
5. 单元素并行批次直接执行，不创建无意义的并发任务。

并行批次使用独立 asyncio Task。普通失败和单工具 timeout 已经是 ToolExecutionResult，不取消
同批其他工具。发生外部取消或 Registry 基础设施异常时，Runner 取消并
`gather(..., return_exceptions=True)` 回收同批未完成 Task，避免遗留后台执行。

## 6. 结果顺序与关联

并行结果按真实完成顺序追加，不恢复为模型调用顺序。每个结果必须携带原始
`tool_call_id`：

```python
ToolResultMessage(
    tool_call_id=call.id,
    tool_name=call.name,
    content=...,
    is_error=...,
)
```

因此调用和结果的权威关联是 ID，而不是列表位置。ContextBuilder 在下一次 Provider 调用前
保留这些 ID。若未来某个 Provider 强制要求特殊排列，应由该 Provider 的消息适配层处理，
不能改变 Session 中的领域消息语义。

## 7. 错误回流

模型可理解和修正的错误全部形成 `ToolResultMessage(is_error=True)`，随后继续 Agent Loop：

| 情况 | 处理 |
|---|---|
| 工具不存在 | 错误 ToolResult，包含可用工具名称 |
| 参数转换后仍不满足 schema | 错误 ToolResult，不调用工具 |
| 工具抛出普通 Exception | 错误 ToolResult，包含异常类型和文本 |
| 工具返回 `ToolOutput(is_error=True)` | 保留模型视图并标记 `is_error=True`；显式完整结果可先写入 ArtifactStore |
| 工具超过 deadline | `timed_out` 结果，转换为错误 ToolResult，模型继续 |
| 工具正常返回 | 标准化为文本，`is_error=False` |

普通字符串的文本内容不再用于推断成功或失败。即使字符串以 `Error:` 开头，只要工具没有显式
返回错误 ToolOutput，它仍是成功结果；这避免源码、日志或用户数据被误判。SkillCatalog 等可预期
领域错误使用显式 `is_error=True`，工具抛出的异常仍由 Registry 归一化。

ContextBuilder 保证结构化错误以明确 `Error:` 文本进入模型上下文。Tool Contract 要求模型把
工具结果视为观察信息，因此模型可以修正参数、选择其他工具或向用户解释失败。

以下情况不包装成普通工具错误：

- `asyncio.CancelledError`：沿工具执行栈传播，由 Runner 统一形成 `cancelled` 结果；
- Registry schema 损坏、Runner 不变量失败等编程或基础设施错误：直接传播，不伪装成模型
  可以解决的业务错误。

timeout 通过 `asyncio.timeout()` 实现。超时时 Python Task 会收到取消；工具若持有子进程、文件或
网络资源，必须在 `CancelledError` / `finally` 中合作式清理。`timed_out` 只表示 Harness 不再
等待，不证明外部操作没有发生。当前阶段只接入内部工具；未来副作用工具必须结合 checkpoint
和 unknown-side-effect 保护，不能把 timeout 当作安全重放依据。

## 8. 不变量

1. 每个被接受执行的 ToolCall 最终至多产生一个同 ID ToolResult；
2. 普通工具错误不终止 Agent Loop；
3. 参数校验失败时工具实现不得被调用；
4. sequential 调用不能和任何相邻调用并行；
5. 并行结果不依赖原始位置，只依赖 `tool_call_id`；
6. 取消不能作为普通错误消息交给模型继续尝试；
7. ToolRegistry 不决定 Run 是否完成，Runner 不自行解释工具业务结果。
8. 单工具 timeout 不取消同批其他正常工具；
9. 外部取消必须取消并回收整个活跃批次；
10. 模型视图与完整结果分离时，完整结果必须成功保存后才能返回 completed 或 reported_error；
11. 工具错误不能通过结果文本前缀推断。
12. timeout 不等价于“工具没有产生副作用”。

## 9. 已验证行为

当前测试覆盖：

- schema 参数转换、required、额外参数、范围和数组限制；
- 未知工具与无效参数回流模型；
- 工具异常与工具主动错误回流模型；
- 并行安全工具确实并发执行；
- 并行结果按完成顺序进入下一次模型请求；
- sequential 工具阻断前后并行批次；
- 工具执行取消形成 cancelled Run。

Execution Policy v0.2 已实现并覆盖：

- 工具级 timeout 覆盖 Registry 默认值；
- timeout 形成稳定 timed_out 结果并回流模型；
- 并行批次中一个工具 timeout 不影响其他工具；
- 外部取消回收所有并行 Task；
- Registry 统一归一化普通值、异常和工具主动错误。

## 10. 暂缓能力

- approval workflow；
- sandbox / remote executor；
- 工具幂等键和外部副作用恢复；
- 全局资源锁和跨进程并发控制；
- 混合依赖 DAG；
- 完整 JSON Schema Draft 支持。
