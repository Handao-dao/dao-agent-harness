# Skill 系统设计

> 状态：Implemented v0.2  
> 最后更新：2026-08-14  
> 范围：SkillCatalog、渐进式加载、ToolRegistry 接入与上下文保留

## 1. 设计结论

DAO 不增加模型无法识别的 Skill 消息角色。Skill 正文通过合法的 ToolCall/ToolResult 进入
Session，但 Harness 通过内部 metadata 将其识别为会话期指令，并给予高于普通工具结果的
上下文保留优先级。

```text
SkillCatalog 元数据
        ↓
ContextBuilder 注入可用目录
        ↓
模型调用 activate_skill
        ↓
ToolRegistry 执行 Skill 工具
        ↓
SkillCatalog.load()
        ↓
ToolResultMessage(kind=skill_instruction)
        ↓
Session 持久化 / ContextGovernor 保护 / 压缩后重新注入
```

消息传输形式与上下文生命周期彼此独立：Provider 只看到标准 tool 消息，DAO 内部则利用
metadata 完成去重、预算和恢复。

## 2. Skill 目录与最小协议

默认扫描：

```text
<workspace>/.dao/skills/<name>/SKILL.md
<user-home>/.dao/skills/<name>/SKILL.md
```

来源优先级为 `workspace > user > builtin`。当前 CLI 接入 workspace 和 user 两级；builtin
保留在 SkillRoot 协议中，等内置 Skill 出现后再注册实际目录。

最小 frontmatter：

```yaml
---
name: pdf
description: Read, create, inspect, and modify PDF documents.
---
```

- `name`、`description` 必填；
- name 只能使用小写字母、数字和单连字符，并与目录名一致；
- description 是激活前的路由协议，默认最多 500 字符；触发与排除条件不能只写在正文；
- 文件按严格 UTF-8 读取；
- 一个损坏 Skill 不阻断 Catalog，其错误进入 diagnostics；
- content hash 由完整 SKILL.md 的 SHA-256 计算，不要求手写版本；
- v0.1 不实现 requires、依赖安装、自定义入口或 Package 管理。

## 3. SkillCatalog

SkillCatalog 是内部只读领域服务，不面向模型，也不维护“已激活”状态：

```python
discover() -> tuple[SkillDescriptor, ...]
get(name) -> SkillDescriptor | None
load(name) -> SkillActivation
resolve_resource(name, relative_path) -> Path
read_resource(name, relative_path, offset, limit) -> SkillResource
refresh() -> None
```

`resolve_resource()` 只接受相对路径，并在解析符号链接后再次确认结果仍位于 Skill 根目录内。
Catalog 的程序化接口可以被 CLI、ContextBuilder 或未来管理命令直接复用，不需要伪造工具调用。

## 4. ToolRegistry 接入

两个模型能力都是标准 `parallel_safe` 工具：

```text
activate_skill(name)
read_skill_resource(skill_name, path, offset?, limit?)
```

工具本身只负责把模型参数适配到 SkillCatalog。参数 schema、call ID、timeout、并行调度、异常
归一化仍由 ToolRegistry 和 Runner 负责。

### 4.1 activate_skill

成功结果使用 `<skill name="..." source="..." location="skill://.../SKILL.md">` 包裹完整
SKILL.md 正文，并携带 `kind=skill_instruction`、skill_name、source、content_hash 和
`retention=session` metadata。source 让日志和未来管理入口可以区分 workspace、user 与 builtin
来源，但不新增模型消息类型。

Skill 指令禁止被 ArtifactStore 自动外置，否则模型只会看到预览而没有完整工作协议。单个正文
使用独立字符上限，超限返回 `skill_too_large`，要求作者把细节拆入 references。

### 4.2 read_skill_resource

资源工具只读取有界 UTF-8 文本片段，返回 MIME、总大小、offset、next_offset 和 eof。其结果
metadata.kind 为 `skill_resource`，属于普通工具观察，可以被截断、外置和摘要。二进制或非 UTF-8
内容不编码为大段 Base64，而是明确要求未来的媒体工具处理。脚本读取与脚本执行保持分离。

### 4.3 失败

SkillCatalog 的可预期错误由工具包装为 `ToolOutput(is_error=True)`，其 content 同时保留稳定的
`Error: [code]` 模型视图，因此 ToolRegistry 会形成
`reported_error` ToolExecutionResult，Runner 再把它作为 `ToolResultMessage(is_error=True)` 回流
模型。稳定错误码包括 skill_not_found、skill_invalid、skill_read_failed、skill_too_large 和
skill_resource_error。

## 5. ContextBuilder

System Prompt 顺序升级为：

```text
Identity
→ Bootstrap Files
→ Tool Contract
→ Available Skills（存在时）
→ Archived Conversation Context（存在时）
→ Extra System Sections
```

Available Skills 只包含转义后的 name 和 description，并指导模型先调用 activate_skill，再按需调用
read_skill_resource。绝对路径、SKILL.md 正文和资源清单不常驻 System Prompt。

Tool Contract 明确规定 Skill 只是任务级执行协议，不能覆盖系统约束、用户意图、授权边界或工具
安全策略。详细的包结构和撰写规则见 [DAO Skill Authoring Guide](skill-authoring-guide.md)。这些
规则提高 Skill 的路由与执行质量，不改变 Catalog → activate_skill → ToolResult 的导入流程。

当 ContextSummary 覆盖了历史 Skill 调用时，ContextBuilder 从被覆盖的强类型消息中派生合法的
Skill ToolCall/ToolResult 前缀。派生前缀只用于 Provider 请求，不写回 Session，也不改变
save_cursor。

## 6. 去重、保留与预算

- 活跃身份以 skill_name 为准，同名 Skill 只保留最近一次激活；
- content_hash 用于识别正文版本，并随 ToolResult 持久化；
- 去重会同时移除被淘汰 ToolResult 及其对应 ToolCall；
- 一个 AssistantMessage 包含多个调用时，只删除对应调用，不破坏其他工具链；
- Skill ToolResult 不参与普通首尾截断或工具 micro-compaction；
- emergency snip 把活跃 Skill 的调用和结果作为一个合法保护块；
- 压缩后的恢复前缀与未覆盖尾部一起再次去重，因此尾部的新激活覆盖旧前缀；
- v0.1 使用确定性的字符预算作为 Skill 专项硬保护，完整请求仍由 ContextGovernor 的 token
  budget 做最终判定。

Session 是事实来源；DAO 不新增 SkillActivated Event、SkillMessage 或可变 active-skills 状态。
压缩、裁剪和恢复都是由 ToolResult metadata 派生的模型视图。

## 7. 暂缓能力

- Skill 安装、更新和卸载工具；
- Package 与通用 ResourceLoader；
- requires 与依赖解析；
- 二进制和多模态资源读取；
- 安全脚本执行器；
- 基于任务相关性的 Skill 淘汰；
- Provider-native token 级 Skill 独立配额；
- 跨 Session 永久激活。
