# 🐈 ClawPhD

一个面向科研的 OpenClaw Agent，能够将学术论文转化为出版级图表、海报、视频等多种形式。本项目基于 OpenClaw 的轻量版本 [Nanobot](https://github.com/HKUDS/nanobot) 构建。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

[English](README.md)

## 功能

- [x] **图表生成** — 根据论文章节自动生成出版级学术插图与统计图表
- [x] **参考图提取** — 从有影响力的论文中搜索并提取真实图表，导出为可编辑的 SVG + PPTX
- [ ] **论文发现** — 主动搜索热门 AI 论文并定时汇总推送
- [ ] **视频讲解** — 根据论文内容生成讲解视频
- [x] **论文网站** — 将论文转化为交互式网页
- [ ] **海报生成** — 根据论文生成学术会议海报
- [ ] **代码生成** — 从论文方法论中提取并生成可复现代码

## 快速开始

### 1. 安装

```bash
# 从源码安装
uv pip install -e .

# 或从 PyPI 安装
pip install clawphd-ai
```

### 2. 初始化

```bash
clawphd onboard
```

这会创建配置文件 `~/.clawphd/config.json` 和默认工作区 `~/.clawphd/workspace/`。

### 3. 配置 API Key

编辑 `~/.clawphd/config.json`，添加至少一个 LLM 提供商的 API Key：

```jsonc
{
  "providers": {
    // 选择一个或多个：
    "openrouter": { "apiKey": "sk-or-..." },
    "anthropic":  { "apiKey": "sk-ant-..." },
    "openai":     { "apiKey": "sk-..." },
    "gemini":     { "apiKey": "AI..." },
    "deepseek":   { "apiKey": "sk-..." }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5"
    }
  }
}
```

如需使用 PaperBanana 图表生成功能，还需设置 Replicate Token：

```bash
export REPLICATE_API_TOKEN="r8_..."
```

### 4. 开始对话

```bash
# 单条消息
clawphd agent -m "你好！"

# 交互式对话
clawphd agent
```

## CLI 命令参考

| 命令 | 说明 |
|---|---|
| `clawphd onboard` | 初始化配置和工作区 |
| `clawphd agent [-m MSG]` | 与 Agent 对话（不加 `-m` 进入交互模式） |
| `clawphd gateway [-p PORT]` | 启动多渠道网关服务 |
| `clawphd status` | 查看配置、API Key 和工作区状态 |
| `clawphd channels status` | 查看渠道连接状态 |
| `clawphd channels login` | 通过扫码连接 WhatsApp |
| `clawphd cron list` | 列出定时任务 |
| `clawphd cron add` | 添加定时任务（`--every`、`--cron` 或 `--at`） |
| `clawphd cron remove <ID>` | 删除定时任务 |
| `clawphd cron enable <ID>` | 启用 / `--disable` 禁用任务 |
| `clawphd cron run <ID>` | 手动触发任务 |

## 支持的 LLM 提供商

通过 [LiteLLM](https://github.com/BerriAI/litellm) 支持以下提供商：

| 提供商 | 配置字段 | 备注 |
|---|---|---|
| OpenRouter | `openrouter` | 聚合多个模型的统一入口 |
| Anthropic | `anthropic` | Claude 系列 |
| OpenAI | `openai` | GPT 系列 |
| Google Gemini | `gemini` | Gemini 系列 |
| DeepSeek | `deepseek` | DeepSeek 系列 |
| Groq | `groq` | 高速推理 |
| 智谱 AI | `zhipu` | GLM 系列 |
| 阿里云通义千问 | `dashscope` | Qwen 系列 |
| Moonshot / Kimi | `moonshot` | Kimi 系列 |
| vLLM | `vllm` | 自部署本地模型 |

## 聊天渠道

启动网关后，在配置中启用所需渠道：

```bash
clawphd gateway
```

| 渠道 | 配置字段 | 说明 |
|---|---|---|
| Telegram | `channels.telegram` | 通过 Bot Token 连接 |
| WhatsApp | `channels.whatsapp` | 通过 Node.js Bridge 连接，需扫码登录 |
| Discord | `channels.discord` | 通过 Bot Token 连接 |
| 飞书 / Lark | `channels.feishu` | 通过 WebSocket 长连接 |

所有渠道均支持 `allowFrom` 白名单控制访问权限。

## Figure Reference 参考图提取

无需额外 API Key，Agent 可自动完成以下流程：

1. 通过 Semantic Scholar 搜索指定话题的高引论文
2. 下载 PDF 并提取所有标注图（Figure N / Fig. N）
3. 将图片保存为 PNG + SVG，并打包为带封面的 PPTX

```bash
clawphd agent -m "帮我找3篇 on-device MoE inference 的论文，把架构图和流程图提取出来做成 PPT"
```

输出文件位于 `<workspace>/outputs/figure_refs/`：

```
outputs/figure_refs/
├── <paper_id>/          ← 每篇论文的 PNG + SVG + paper.pdf + figures.csv
└── reference_pack_<topic>_<ts>.pptx
```

支持按图片类型筛选（架构图、实验图、概念示意图等）。若配置了视觉模型，还可用 VLM 对图片做精细分类。

---

## PaperBanana 图表生成

配置 VLM / 图像生成提供商后，Agent 将自动获得三个图表工具：

| 工具 | 功能 |
|---|---|
| `search_references` | 搜索参考图库，用于上下文学习 |
| `generate_image` | 生成方法论框架图或统计图表 |
| `critique_image` | 评价图表质量并给出修改建议 |

支持两种图表类型：
- **方法论框架图** — 通过图像生成模型（Gemini / Replicate）渲染
- **统计图表** — 通过 VLM 生成 matplotlib 代码后执行

### 参考图库配置

将参考图库放在以下任一位置（按顺序查找）：

1. `<项目根目录>/paperbanana/data/reference_sets/` — 开发环境
2. `~/.clawphd/references/` — 安装后使用

目录中需包含 `index.json`：

```json
{
  "examples": [
    {
      "id": "ref_001",
      "source_context": "...",
      "caption": "...",
      "image_path": "images/ref_001.jpg",
      "category": "flow"
    }
  ]
}
```

## 许可证

[MIT](LICENSE)
