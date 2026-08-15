# DAO Skill Authoring Guide

> 状态：Active v0.1  
> 最后更新：2026-08-14  
> 范围：面向 DAO Agent Harness 的 Skill 包结构、路由描述、正文与资源组织

## 1. Skill 的职责

Skill 是按需加载的任务执行协议，不是新的消息角色，也不是绕过 ToolRegistry 的执行通道。
Catalog 先向模型暴露名称和描述；模型确认适用后调用 `activate_skill`，正文以标准 ToolResult
进入上下文。Skill 引用的文本资料再由 `read_skill_resource` 分段加载；脚本执行、二进制读取和
其他副作用仍必须经过对应工具的权限与安全边界。

Skill 指令服从以下优先级：系统约束、用户意图、授权边界与工具安全策略始终高于 Skill 正文。
Skill 只能细化当前任务的做法，不能自行扩大权限或任务范围。

## 2. 最小目录

```text
<skill-name>/
├─ SKILL.md                 # 必需：路由元数据与主工作流
├─ scripts/                 # 可选：确定性、可复用的程序
├─ references/              # 可选：按需读取的详细知识
└─ assets/                  # 可选：用于产出的模板、图片或其他素材
```

- 不要在 Skill 内附加 README、安装说明或 changelog；对模型有用的内容应进入 SKILL.md 或
  references；
- `scripts/` 中的文件不会因为激活 Skill 自动执行；
- `assets/` 默认不会注入模型上下文，应由合适的资源或媒体工具使用；
- DAO v0.1 不读取其他运行时专有的 manifest 或代理配置文件。

## 3. Frontmatter 是路由协议

SKILL.md 必须是 UTF-8，并只使用 DAO 当前支持的字段：

```yaml
---
name: pdf-inspection
description: Inspect PDF structure, extract bounded text, and verify rendered pages. Use for PDF reading, validation, or layout QA; do not use for general image editing or slide decks.
---
```

### 3.1 name

- 使用小写字母、数字和单连字符；
- 必须与目录名一致；
- 推荐使用简短、动词或任务导向的名称；
- 避免把实现技术写进名称，除非它本身决定使用场景。

### 3.2 description

`description` 是模型激活 Skill 之前唯一可见的路由依据，必须同时回答：

1. 这个 Skill 能完成什么；
2. 哪些用户意图、文件类型或任务信号应触发它；
3. 哪些相近场景不应触发它。

所有 “何时使用” 信息都应写在 description 中，不能只放在正文的 `When to Use` 章节。
DAO 默认限制 description 不超过 500 个字符；优先提高区分度，不要堆砌同义关键词。

## 4. SKILL.md 正文

正文只在 Skill 被激活后加载，因此它负责执行，而不负责首次路由。推荐采用祈使语气并围绕
可验证工作流组织：

```markdown
# PDF Inspection

## Workflow
1. 检查输入和目标产物。
2. 读取需要的参考资料。
3. 执行转换或分析。

## Verification
- 渲染关键页面并检查布局。
- 核对页数、文本和输出路径。

## Failure handling
- 缺少输入时停止并说明缺失项。
- 工具失败时保留原始错误，不宣称任务成功。

## Resources
- 详细格式规则见 `references/pdf-layout.md`。
- 确定性检查使用 `scripts/inspect_pdf.py`。
```

- 只写模型无法可靠推断的流程、约束和领域知识；
- 不重复系统提示、Tool Contract 或工具 schema 已经表达的规则；
- 把易错且固定的步骤写得具体，把需要判断的步骤保留合理自由度；
- 正文保持在 500 行以内；细节拆到 references，避免主工作流被背景知识淹没；
- 同一信息只保留一个权威位置，避免正文与参考资料逐渐不一致。

## 5. 资源分层

### 5.1 scripts

脚本适合确定性高、重复发生、靠自然语言容易出错的工作，例如格式转换、校验和机械化提取。
正文应说明何时使用、参数、预期输出和验证方式。脚本仍通过 DAO 提供的执行工具运行，不能把
“文件存在”当成“已经获准执行”。

### 5.2 references

参考资料适合格式规范、长示例、领域表格和边缘情况。SKILL.md 应直接指向需要读取的文件，尽量
保持一层引用，避免模型连续追踪多级链接。较长文档应提供目录或可搜索标题；读取时使用 offset
与 limit 分段获取相关部分。

### 5.3 assets

素材是产出资源，不是默认提示词，例如文档模板、图标和样例文件。正文应说明它们的用途和预期
产物，但不要要求把大型或二进制素材直接展开进文本上下文。

## 6. 自检清单

发布或更新 Skill 前至少检查：

- 目录名与 name 一致，文件为 UTF-8；
- description 同时覆盖能力、正向触发和相近的排除场景；
- 只看 Catalog 中的 name + description 也能做出正确路由判断；
- 正文从执行步骤开始，没有依赖尚未读取的隐含资料；
- 每个引用路径真实存在，且没有越出 Skill 目录；
- 固定步骤有明确验证，失败路径不会伪造成功；
- scripts、references 与 assets 没有重复保存同一份说明；
- Skill 没有试图覆盖系统约束、用户授权或工具策略。

建议为重要 Skill 保存一组正向、反向和近似意图样例。DAO 暂不引入 SkillMatcher 或自动路由
评分器；这些样例先用于人工与端到端模型评测，不改变当前 Catalog → activate_skill 流程。
