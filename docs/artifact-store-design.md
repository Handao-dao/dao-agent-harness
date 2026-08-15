# ArtifactStore 设计

> 状态：Implemented v0.2  
> 日期：2026-08-14  
> 范围：大型工具结果外置、稳定引用、受控回读，以及与 Session/Context 的边界

## 1. 问题与定位

工具可能返回日志、网页、文件或命令输出等大型文本。如果完整结果直接写入
`ToolResultMessage.content`，它会同时膨胀：

- Session Event Log；
- Entry Tree 内存视图；
- 每次发给 Provider 的上下文；
- 分支会话继承的历史。

ContextGovernor 可以缩短临时 Provider 视图，但不会改变 Session 中保存的原始结果。因此需要独立的
ArtifactStore：将大型结果保存为共享对象，消息只保留可读预览和稳定引用。

ArtifactStore 属于 Tool Runtime 的结果加工层，不负责普通附件、Memory、ContextSummary 或任务恢复
Checkpoint。第一版只处理 UTF-8 文本工具结果。

## 2. nanobot 参考与 DAO 调整

nanobot 在 Runner 归一化工具结果时检查长度，超限后写入 workspace 文件，并把本地文件路径和前缀
预览返回给模型。这个机制证明了“先外置，再让模型按需读取”的可行性。

DAO 保留这一思想，但调整三点：

1. 不把宿主机路径暴露给模型，消息只出现逻辑 Artifact ID；
2. 使用内容寻址，使相同内容天然去重，也能被不同 Session/分支安全共享；
3. 把外置策略放在 ToolRegistry 的执行边界，而不是 Runner，保持 Runner 只负责批次调度。

## 3. ArtifactRef

```python
@dataclass(frozen=True, slots=True)
class ArtifactRef:
    id: str
    media_type: str
    size_bytes: int
    size_chars: int
    sha256: str
```

约束：

- `id` 格式为 `art_<64 位小写 sha256>`；
- SHA-256 基于原始文本的严格 UTF-8 字节计算；
- `media_type` 第一版固定为 `text/plain; charset=utf-8`；
- 引用不包含 Session、Run、Tool Call 或物理路径；
- `ArtifactRef` 是值对象，可以安全复制到 Entry Tree 的多个分支。

回读结果使用：

```python
@dataclass(frozen=True, slots=True)
class ArtifactSlice:
    ref: ArtifactRef
    content: str
    offset: int
    next_offset: int
    eof: bool
```

`offset`、`next_offset` 均按 Python 字符索引解释；`size_bytes` 只用于展示和完整性验证。

## 4. ArtifactStore 协议

```python
class ArtifactStore(Protocol):
    async def put_text(self, content: str) -> ArtifactRef: ...

    async def read_text(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        limit: int = 4000,
    ) -> ArtifactSlice: ...
```

第一版提供两个实现：

- `InMemoryArtifactStore`：单元测试和嵌入式运行；
- `LocalArtifactStore`：CLI 默认持久化实现。

协议只接受 Artifact ID，不接受任意路径，避免把 Artifact 工具变成文件系统读取旁路。

## 5. LocalArtifactStore

默认目录位于 CLI 数据目录下的 `artifacts/`。物理布局为：

```text
<root>/<sha256 前两位>/<完整 sha256>.txt
```

写入流程：

1. 对文本执行严格 UTF-8 编码并计算 SHA-256；
2. 若目标已存在，校验后直接复用；
3. 否则写入同目录临时文件；
4. flush、`fsync` 后原子 replace；
5. 返回 ArtifactRef。

读取时必须验证 ID 格式，拒绝路径分隔符和非规范 ID；随后验证 UTF-8、字节长度与 hash。损坏、缺失和
非法引用使用明确的 ArtifactStore 异常，不把底层路径泄露给模型。

## 6. ArtifactPolicy

```python
@dataclass(frozen=True, slots=True)
class ArtifactPolicy:
    externalize_above_chars: int = 16_000
    preview_head_chars: int = 2_000
    preview_tail_chars: int = 2_000
    read_chunk_chars: int = 4_000
```

第一版规则：

- 只外置成功且超过阈值的文本结果；
- 普通失败、timeout、cancelled 和 validation error 不外置；
- 预览同时保留头部与尾部；
- 边界按字符计算，避免切断 Unicode 字符；
- `read_artifact` 单次读取上限不得高于外置阈值，防止回读结果再次被外置。

外置后的模型可见内容使用稳定模板：

```text
[tool result externalized]
artifact_id: art_<sha256>
media_type: text/plain; charset=utf-8
size: <chars> chars / <bytes> bytes
Use read_artifact with artifact_id and optional offset/limit to inspect more.

--- head preview ---
...
--- omitted ---
--- tail preview ---
...
```

预览仍然是不可信工具数据，不能被解释为系统指令。

## 7. ToolExecutionResult 与 ToolResultMessage

两个类型都增加：

```python
artifact_refs: tuple[ArtifactRef, ...] = ()
```

第一版一次结果最多产生一个引用，但使用 tuple 保留多附件和多模态扩展空间。

未适配工具的大结果继续走通用 fallback：

```text
Tool.execute
  → Registry 归一化文本
  → ArtifactPolicy 判断
  → ArtifactStore.put_text
  → ToolExecutionResult(content=预览, artifact_refs=(ref,))
  → Runner 按原 tool_call_id 构造 ToolResultMessage
  → SAVE 写入 Session
```

理解自身结果语义的工具可以直接分离模型视图与完整内容：

```text
ToolOutput(content=简洁模型视图, artifact_content=完整原文)
  → ArtifactStore.put_text(artifact_content)
  → Registry 在 content 后追加 artifact_id 和读取说明
  → ToolExecutionResult(content=模型视图+引用, artifact_refs=(ref,))
```

显式 `artifact_content` 不受长度阈值影响。如果 Registry 没有配置 ArtifactStore，或保存、完整性
校验失败，则返回 `failed/artifact_store`；不会只交付摘要却伪装成完整成功。

工具主动报告的失败也可以提供 `artifact_content`。保存成功后仍返回
`failed/reported_error`，同时携带 ArtifactRef；保存失败则返回 `failed/artifact_store`。普通错误文本
不会自动外置，只有工具明确声明的完整失败输出才进入 ArtifactStore。

ContextBuilder 第一版仍只把 `content` 发给 Provider。`artifact_refs` 保留在强类型 Session 中，用于
持久化、UI 展示和未来 Provider 原生附件适配。

ContextGovernor 替换工具结果内容时必须保留原有 `artifact_refs`；补链占位消息使用空引用。

## 8. ToolRegistry 接入

ToolRegistry 增加可选依赖：

```python
artifact_store: ArtifactStore | None
artifact_policy: ArtifactPolicy
```

执行顺序固定为：参数校验 → 工具执行/timeout → 结果归一化 → 工具专属 Artifact 或通用外置 → 返回
`ToolExecutionResult`。

ArtifactStore 写入发生在 Session SAVE 之前，因此：

- SAVE 成功时，消息引用一定已经可读；
- SAVE 失败时可能留下无引用对象，但不会产生已提交的悬空引用；
- 第一版接受孤儿对象，不在热路径执行删除。

如果完整工具结果无法写入 ArtifactStore，不得退回并持久化完整原文。Registry 返回失败结果：

- `error_code="artifact_store"`；
- `content` 只包含稳定错误说明和受限头尾预览；
- `artifact_refs=()`。

对于工具主动提供的模型视图，失败内容保留该有界视图并追加稳定的“完整结果不可用”说明，不把
`artifact_content` 的片段泄漏回上下文；通用 fallback 则继续提供受限头尾预览。

这表示“工具执行成功，但结果无法可靠交付”，不证明工具可安全重放。模型可以解释失败或改用其他
工具，Runner 不自动重试。

## 9. read_artifact 工具

提供内建 `ReadArtifactTool`：

- 名称：`read_artifact`；
- 参数：`artifact_id`、可选 `offset`、可选 `limit`；
- `parallel_safe=True`；
- 只通过 ArtifactStore 回读，不接受路径；
- `limit` 由工具再次限制到 `read_chunk_chars`；
- 返回正文、区间、`next_offset` 和 `eof`，便于模型分页读取；
- 使用 `ToolOutput(allow_externalization=False)`，避免分页结果递归外置。

当 Registry 配置了 ArtifactStore 时，CLI 自动注册该工具；未配置时不暴露。其输出天然小于外置
阈值，不进入递归外置。

## 10. Session Codec 与 Entry Tree

`ToolResultMessage.artifact_refs` 属于 Session 事实，必须持久化。当前 Session Codec schema v6：

- 编码时写入 `artifact_refs`；
- 读取 v1-v5 时兼容旧消息，缺少引用时默认空 tuple；
- 不回写旧记录；
- JSONL Session Event Log 的事件封装版本保持不变，因为变化位于可向后兼容的 Message payload。

Entry Tree 不复制 Artifact 内容，只复制引用。不同分支和 Session 可以指向同一对象，因此 `/clear`
只清理当前 Session，不删除 Artifact。

## 11. 安全与一致性

- ID 必须完整匹配规范格式，禁止路径遍历；
- Artifact 内容始终按不可信数据处理；
- 不把本地绝对路径、临时文件名或底层异常细节发送给模型；
- 写后读必须校验内容 hash；
- ArtifactStore 不负责判断或过滤敏感信息，调用方仍需避免把凭据交给工具结果；
- 第一版不提供删除 API，避免共享引用被误删。

## 12. 第一版实施顺序

1. 定义 ArtifactRef、ArtifactSlice、ArtifactPolicy 和异常；
2. 实现 InMemoryArtifactStore 与 LocalArtifactStore；
3. 扩展 ToolExecutionResult 和 ToolResultMessage；
4. Session Codec 持久化 ArtifactRef 并补迁移测试（当前 schema v6）；
5. Registry 接入成功大结果外置和失败降级；
6. 实现并注册 ReadArtifactTool；
7. CLI 增加 Artifact 目录配置；
8. 补去重、分页、损坏检测、分支共享和 SAVE 失败测试。

## 13. 暂缓事项

- 引用扫描与垃圾回收；
- 二进制和多模态 Artifact；
- 压缩、加密与远端对象存储；
- ACL、租户隔离和容量配额；
- GUI Artifact 卡片、下载和预览；
- Memory、ContextSummary 与 Artifact 的统一检索层。

这些能力都建立在稳定 ArtifactRef 之上，不阻塞第一版 CLI Harness。
