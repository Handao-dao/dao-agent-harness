# 基础工具设计

> 状态：Implemented v0.3（七个基础工具已加入）  
> 最后更新：2026-08-14  
> 参考：pi coding-agent 的 `read`、`write`、`edit`、`bash`、`grep`、`find`、`ls`

## 1. 设计结论

DAO 借鉴 pi 已验证的工具参数、模型输出和错误语义，但不复制其 TypeScript、TUI、扩展钩子或
自动下载外部命令等产品实现。基础工具继续实现现有 `AgentTool` 协议，并复用
`ToolRegistry`、`ToolOutput` 和 `ArtifactStore`：

```text
模型参数
  → ToolRegistry schema 转换与校验
  → 基础工具执行
  → 工具生成有界、可继续操作的 model view
  → 可选完整 artifact_content
  → ToolExecutionResult / ToolResultMessage
```

第一组工具固定为：

| 工具 | 目标 | 并发标签 | 状态 |
|---|---|---|---|
| `read` | 分页读取 UTF-8 文本 | `parallel_safe` | 已实现 |
| `ls` | 稳定列出目录项 | `parallel_safe` | 已实现 |
| `find` | 按 glob 查找路径 | `parallel_safe` | 已实现 |
| `grep` | 搜索文件内容 | `parallel_safe` | 已实现 |
| `write` | 新建或完整覆写文件 | `sequential` | 已实现 |
| `edit` | 原子精确替换一个文件 | `sequential` | 已实现 |
| `bash` | 执行可取消的 Shell 命令 | `sequential` | 已实现 |

第一版不试图表达“同文件串行、不同文件并行”的资源锁，因此修改工具对 Runner 声明为
`sequential`。内部仍提供文件级 Mutation Queue，为未来细化并发策略保留正确边界。

## 2. 共享基础设施

### 2.1 ToolPathPolicy

所有文件工具使用同一条路径解析规则：

1. 工具构造时注入 workspace，模型不能修改；
2. 相对路径基于 workspace 解析；
3. `~` 和绝对路径规范化后仍必须位于 workspace；
4. 默认拒绝 `..`、绝对路径或符号链接造成的 workspace 逃逸；
5. 只有宿主显式设置 `allow_outside_workspace=True` 时才允许外部路径；
6. 模型可见路径优先使用 `/` 分隔的 workspace 相对表示。

这比 pi 默认的自由绝对路径更适合作为 Harness 基线。未来 approval 或 sandbox 可以改变授权策略，
但不需要修改每个工具的路径实现。

### 2.2 TruncationResult

共享截断同时计算完整行数和 UTF-8 字节数，默认限制继承 pi：

```text
max_lines = 2000
max_bytes = 50 * 1024
grep_max_line_chars = 500
```

- `read`、`ls`、`find`、`grep` 使用 head truncation；
- `bash` 使用 tail truncation；
- head truncation 不返回半行；
- tail truncation 只在单行本身超过字节预算时允许一个 UTF-8 安全的边界半行；
- 结果记录 `truncated_by`、原始/输出行数、原始/输出字节数和实际限制；
- 工具必须把可执行的下一步写进模型视图，不能只写“内容已截断”。

### 2.3 FileMutationQueue

`write` 和 `edit` 将以规范绝对路径作为锁键：同一路径严格串行，不同路径可以同时进入工具内部。
锁只约束当前进程，不宣称提供跨进程或分布式一致性。

## 3. read

参数协议：

```json
{
  "path": "src/example.py",
  "offset": 1,
  "limit": 200
}
```

- `path` 必填；
- `offset` 可选，使用从 1 开始的行号；
- `limit` 可选，先限制候选行范围，再应用统一行数/字节预算；
- 只接受 UTF-8 文本，首版不把图片或二进制偷偷转换为文本；
- 自动识别 UTF-8 BOM，模型视图统一使用 LF；
- 空文件返回 `(empty file)`；
- offset 越界、目录、缺失文件、非 UTF-8 文件形成普通工具错误并回流模型；
- 截断或用户 limit 提前停止时返回准确 `next_offset`；
- 单行超过 50KB 时不输出半行，等待后续 `bash` 或专用分片读取能力处理。

`read` 返回的是已经治理过的模型视图，并设置 `allow_externalization=False`。它不把文件复制到
ArtifactStore：原文件已经是完整权威内容，模型通过下一 offset 继续读取。Registry 因此不会对
续读提示执行第二次通用头尾截断。

metadata 只供 Harness 和未来 UI 使用：

```python
{
    "kind": "file_read",
    "path": "src/example.py",
    "offset": 1,
    "next_offset": 201,
    "total_lines": 820,
    "truncation": {...},
}
```

CLI 默认注册 `read`，workspace 来自 `--workspace`。

## 4. ls

参数协议：

```json
{
  "path": ".",
  "limit": 500
}
```

- `path` 可选，默认为 workspace 根目录；
- `limit` 可选，默认 500，必须是正整数；
- 使用忽略大小写的稳定排序，并用原名称作为并列排序键；
- 包含 dotfiles，目录名称追加 `/`；
- 空目录返回 `(empty directory)`；
- 达到数量上限时提示模型扩大 limit，达到 50KB 上限时建议收窄目录；
- 无法读取类型的单个目录项会被跳过并记录数量；
- 返回 `directory_listing` metadata，包括相对路径、返回/总条目数、数量限制和截断统计；
- 与 `read` 一样设置 `allow_externalization=False`，防止已经有界的结果被二次外置。

CLI 默认注册 `ls`，并与 `read` 共享 `--workspace` 权限边界。

## 5. 其余工具实现

### 5.1 find / grep

- `find`：`pattern/path/limit`，使用 `pathlib` 递归 glob，输出相对 POSIX 路径，默认最多 1000 项；
- `grep`：支持 regex/literal、ignore case、glob、context 和 limit，格式为
  `path:line: content`，默认最多 100 个匹配；
- 二者跳过 `.git`、`node_modules` 和 `__pycache__`，但暂不实现完整 `.gitignore` 语义；
- 达到结果上限时提示模型扩大 limit 或收窄条件。

### 5.2 write / edit

- `write` 只用于新文件或完整覆写，自动创建父目录；
- `edit` 接受一个 `edits[]`，所有 `oldText` 都针对同一份原文件匹配；
- `oldText` 必须非空且唯一，各修改范围不得重叠；
- 任意修改失败则整个调用不落盘；
- 保留原始 LF/CRLF 和 UTF-8 BOM；
- 成功结果在 metadata 中携带 unified patch；
- 首版先实现精确匹配，pi 的 Unicode 与尾随空格模糊匹配作为后续兼容层。

### 5.3 bash

- 参数为 `command` 和可选 `timeout`；模型参数不能延长 Harness 配置的最大 deadline；
- 合并 stdout/stderr 并等待命令结束，当前不实现流式增量；
- 取消或 timeout 时终止当前 Shell 子进程，暂不实现完整进程树治理；
- 模型视图保留最后 2000 行或 50KB；
- 截断时把完整输出作为 `artifact_content`，不暴露宿主临时文件路径；
- 非零退出仍是 `is_error=True`，但可以携带 ArtifactRef，确保失败日志可继续检查。

## 6. 错误 Artifact

`ToolOutput(is_error=True, artifact_content=...)` 已被允许。Registry 的顺序为：

```text
保存完整失败输出
  → 成功：failed/reported_error + artifact_refs
  → 失败：failed/artifact_store + 有界模型视图
```

这不会把失败改成成功，只是保证测试日志、编译错误和未来 MCP 错误的完整观察结果可以被模型继续
分页读取。没有显式 `artifact_content` 的普通错误仍不会因为文本很长而自动外置。

## 7. 实施顺序

```text
ToolPathPolicy / truncation / FileMutationQueue  ✓
read                                               ✓
ls                                                 ✓
find → grep                                          ✓
write → edit                                        ✓
bash                                                ✓
```

七个工具当前只作为 Harness 能力骨架：复用 pi 的名称、参数和核心返回方式，由 Python 直接实现。
测试只保留一条基础工具贯穿式冒烟链路；完整 `.gitignore`、模糊编辑、Shell 进程树、流式输出、
approval 和 sandbox 等生产能力留到确实需要使用时再扩展。
