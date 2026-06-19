# Academic Research Skills — Adapted for OpenCode & Hermes

此目录包含 Academic Research Skills (ARS) 4 个核心技能的改写版本，兼容 **OpenCode** 和 **Hermes Agent** 两种平台。

---

## 目录结构

```
adapted/
├── README.md                    # 本文件
├── shared -> ../shared          # 共享资源 symlink
│
├── opencode/                    # OpenCode 布局（扁平）
│   ├── opencode.json            # OpenCode 配置（直接指向当前目录）
│   ├── deep-research/
│   │   ├── SKILL.md             # Hermes frontmatter（兼容双平台）
│   │   ├── agents/   -> symlink
│   │   ├── references/ -> symlink
│   │   ├── templates/ -> symlink
│   │   └── examples/  -> symlink
│   ├── academic-paper/
│   ├── academic-paper-reviewer/
│   └── academic-pipeline/
│
└── hermes/                      # Hermes 布局（分类）
    ├── research/
    │   ├── DESCRIPTION.md       # 分类描述
    │   └── deep-research/
    │       ├── SKILL.md
    │       └── ... (symlinks)
    ├── paper-writing/
    │   ├── DESCRIPTION.md
    │   └── academic-paper/
    ├── paper-review/
    │   ├── DESCRIPTION.md
    │   └── academic-paper-reviewer/
    └── pipeline/
        ├── DESCRIPTION.md
        └── academic-pipeline/
```

---

## 核心改写策略

### 1. SKILL.md Frontmatter

两个平台的核心差异在于 frontmatter。**Hermes 是 OpenCode 的超集**，所以用 Hermes 格式的 frontmatter 可以兼容两者：

| 字段 | OpenCode | Hermes |
|------|----------|--------|
| `name` | 必需 | 必需 |
| `description` | 必需 | 必需 |
| `version` | 忽略 | 推荐 |
| `author` | 忽略 | 推荐 |
| `license` | 可选 | 推荐 |
| `platforms` | 忽略 | 推荐 |
| `metadata.hermes.*` | 忽略 | Hermes 专用标签 |

**改写方法：** 在原有 frontmatter 基础上，补充 Hermes 所需的字段。OpenCode 会静默忽略不认识的字段。

```yaml
# 原始（Claude Code）
---
name: deep-research
description: "..."
metadata:
  version: "2.10.0"
  status: active
---

# 改写后（兼容 OpenCode + Hermes）
---
name: deep-research
title: Deep Research — Universal Academic Research Agent Team
description: "..."
version: 2.10.0
author: Academic Research Skills
license: CC-BY-NC-4.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Research, Academic, ...]
    category: research
    related_skills: [academic-paper, academic-pipeline]
    requires_toolsets: [terminal, files]
---
```

### 2. 目录结构

| 平台 | 布局 | 示例 |
|------|------|------|
| **OpenCode** | 扁平: `<name>/SKILL.md` | `opencode/deep-research/SKILL.md` |
| **Hermes** | 分类: `<category>/<name>/SKILL.md` | `hermes/research/deep-research/SKILL.md` |
| **Claude Code** | 扁平: `skills/<name>/SKILL.md` | `skills/deep-research/SKILL.md` |

两个平台都扫描 `**/SKILL.md` 文件，所以目录层级不影响加载。

### 3. 共享资源

`agents/`、`references/`、`templates/`、`examples/` 等辅助目录通过 **symlink** 指向原始项目目录，避免重复存储。例如：

```bash
# OpenCode 布局中
adapted/opencode/deep-research/agents/ -> ../../skills/deep-research/agents/

# Hermes 布局中
adapted/hermes/research/deep-research/agents/ -> ../../../skills/deep-research/agents/
```

---

## 如何在新 skill 上应用此适配

### 步骤 1: 准备 SKILL.md

```
1. 复制原始 SKILL.md
2. 替换 frontmatter 为 Hermes 兼容格式
3. 添加: version, author, license, platforms, metadata.hermes
4. SKILL.md body 内容不变
```

### 步骤 2: 创建目录结构

```bash
# OpenCode
mkdir -p adapted/opencode/my-skill/
cp /path/to/my-skill/SKILL.md adapted/opencode/my-skill/SKILL.md
# 修正 frontmatter ...

# Hermes
CATEGORY="research"  # 选择合适分类
mkdir -p adapted/hermes/$CATEGORY/my-skill/
cp adapted/opencode/my-skill/SKILL.md adapted/hermes/$CATEGORY/my-skill/SKILL.md
```

### 步骤 3: 链接共享资源

```bash
ln -s /abs/path/to/agents  adapted/opencode/my-skill/agents
ln -s /abs/path/to/references  adapted/opencode/my-skill/references
# ... 同样为 hermes 创建 symlink
```

### 步骤 4: （可选）Hermes 分类 DESCRIPTION.md

如果 Hermes 的目标分类还不存在，创建 `DESCRIPTION.md`：

```yaml
---
description: Skills for your category here.
---
```

### 步骤 5: 使用

**OpenCode 载入：**
```json
{
  "skills": {
    "paths": ["adapted/opencode"]
  }
}
```

**Hermes 载入：**
```bash
cp -r adapted/hermes/* ~/.hermes/skills/
# 或 symlink:
ln -s $(pwd)/adapted/hermes/* ~/.hermes/skills/
```

---

## 注意事项

1. **Hermes 特有字段**：`metadata.hermes.tags`、`category`、`related_skills`、`requires_toolsets` 只对 Hermes 生效，OpenCode 忽略。
2. **`version` 字段**：在 Hermes 中有实际含义（版本追踪），OpenCode 中无副作用。
3. **Body 内容不需要改**：两个平台都读取 markdown body 作为 skill 指令，内容完全一样。
4. **Symlink 优于复制**：辅助目录用 symlink 而非复制，保持与原始项目同步。
