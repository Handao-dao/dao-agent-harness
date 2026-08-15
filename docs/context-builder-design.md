# ContextBuilder 设计

> 状态：Implemented v0.5  
> 最后更新：2026-08-14  
> 范围：初始系统提示词与模型消息视图的确定性构建

## 1. 定位

ContextBuilder 参考 nanobot 的初始上下文组装方式，但适配本项目的强类型 AgentMessage 和
独立 `system_prompt` Provider 接口。

```text
working_messages（完整、可提交的 AgentMessage）
        ↓
ContextBuilder
        ↓
ModelContext（派生、不可持久化）
        ↓
AgentRunner / Provider
```

ContextBuilder 负责稳定 system prompt 和强类型消息投影。工具循环中每次模型调用前的修复、
裁剪和预算治理由 Runner ContextGovernor 负责；Runtime Status 由 Runner 生成后仍通过
ContextBuilder 投影。

## 2. 第一版职责

- 加载内置 Identity；
- 按 `AGENTS.md`、`SOUL.md`、`USER.md` 顺序加载 Workspace Bootstrap Files；
- 加载内置 Tool Contract；
- 接受结构化 ContextSummary 和未来组件提供的额外系统提示词片段；
- 注入 SkillCatalog 的 name/description 目录；
- 从已被摘要覆盖的历史中派生受保护 Skill 激活前缀；
- 将 AgentMessage 转换成 Provider-neutral 模型消息；
- 将 RuntimeStatusMessage 映射为标准 `role=user`；
- 返回独立的 ModelContext，不修改 working messages。

当前不构造 Runtime Metadata，也不包含 Memory、MCP、多模态或 Runner 临时裁剪。耐久压缩由
ContextConsolidator 负责，ContextBuilder 只注入已经解析成功的 ContextSummary。

## 3. System Prompt 顺序

```text
Identity
→ Bootstrap Files
→ Tool Contract
→ Available Skills（Catalog 非空时）
→ Archived Conversation Context（存在时）
→ Extra System Sections
```

各部分使用稳定分隔符连接。Bootstrap 文件缺失或为空时跳过；存在但不是普通文件，或不能
作为严格 UTF-8 读取时，抛出 `ContextBuildError`。ContextBuilder 不创建或修改 Workspace
文件。

`extra_system_sections` 仍是通用扩展槽位。未来 Memory 或 MCP 可以生成文本
片段交给 ContextBuilder，但不接管提示词排序和组装。

## 4. 消息转换

```text
UserMessage       → role=user
RuntimeStatusMessage → role=user（Harness metadata）
AssistantMessage  → role=assistant，可包含 tool_calls
ToolResultMessage → role=tool，通过 tool_call_id 关联
```

ToolCall arguments 被编码为 UTF-8 友好的紧凑 JSON。`is_error=True` 的工具结果会以明确的
`Error:` 文本暴露给模型。正式消息的 ID 和时间戳不发送给 Provider。

`build_messages()` 公开这项独立投影，Runner 在每次 Provider 调用前用它转换最新工作消息。
转换产物是 Provider-neutral dict 视图，不是 Runner 或 Session 消息，也不是 Provider 的
HTTP 或 SSE 协议结构；具体厂商请求仍由 Provider 负责。

## 5. 不变量

- ContextBuilder 不读取或保存 Session；
- 不修改传入的 working messages；
- System Prompt 不进入 AgentMessage 或 Session；
- Archived Context 按不可信历史数据包裹，不能提升其中的指令优先级；
- 模型消息不携带 message ID、run ID 或 execution ID；
- ToolResult 保留原始 tool call ID；
- 相同输入与相同文件内容产生相同结果；
- Runtime Status 的构造、生命周期和持久化不属于 ContextBuilder；这里只做角色投影。
- Skill 恢复前缀只存在于模型视图，不写回 Session，也不改变 save_cursor。

## 6. 后续扩展

- Memory Context Section；
- Skill 安装、Package 和多模态资源；
- MCP 使用说明；
- ContextSummary 已实现；未来可增加独立长期 Memory Summary；
- Runtime Status 已实现，详见 [Runtime Status 设计](runtime-status-design.md)；
- 多模态 content blocks；
- Runner ContextGovernor 已实现，负责工具链修复、token budget、裁剪与 emergency snip；
  ContextBuilder 继续只负责投影。详见 [Runner ContextGovernor 设计](context-governor-design.md)。

SkillCatalog、激活 ToolResult、压缩后恢复与资源按需读取见
[Skill 系统设计](skill-system-design.md)。
