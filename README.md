# GrowthGuard

GrowthGuard 是一个面向 DTC 与电商业务的 AI 增长分析 Agent。

用户可以直接使用自然语言提问，系统会自动识别分析意图、制定分析计划、调用对应的数据分析工具，并生成业务结论和行动建议。

> 本仓库不包含真实企业数据。原始数据与清洗后的生产数据因商业保密要求未公开。

---

## 核心功能

* 自然语言业务问答
* 自动分析规划与工具选择
* 销售、漏斗、订阅、留存、退款、营销和产品分析
* 多轮对话上下文解析
* Session Memory
* Guardrails
* 结构化日志与可观测性
* FastAPI 后端服务
* Streamlit 前端界面
* Docker 容器化运行

---

## 系统流程

```text
用户问题
→ Streamlit
→ FastAPI
→ Context Resolver
→ Planner
→ Analytics Tools
→ Final Response Agent
→ 分析结论与行动建议
```

大语言模型主要负责：

* 理解用户问题
* 制定分析计划
* 选择分析工具
* 解释分析结果

Python 工具负责：

* 读取数据
* 计算业务指标
* 比较时间变化
* 返回可验证的结构化结果

这种设计可以降低数字幻觉，并提高结果的准确性和可追溯性。

---

## 支持的分析模块

| 模块               | 功能                 |
| ---------------- | ------------------ |
| Sales            | 销售趋势与渠道贡献          |
| Funnel           | 网站转化漏斗             |
| Subscription     | 订阅增长、流失与停用         |
| Cohort           | 客户留存分析             |
| Refund           | 退款趋势与压力            |
| Marketing        | Campaign 与 Flow 表现 |
| Product          | 产品和 SKU 表现         |
| Multi-tool Analysis | 组合多个分析工具完成综合增长诊断 |

---

## 技术栈

* Python
* OpenAI Agents SDK
* FastAPI
* Streamlit
* pandas
* Pydantic
* Docker
* Docker Compose
* Session Memory
* Structured Logging

---

## 项目结构

```text
growthguard/
├── api/                    # FastAPI 服务
├── app/                    # Streamlit 前端
├── sources/
│   ├── agent/              # Planner、上下文解析与最终回答
│   ├── tools/              # 数据分析工具
│   ├── memory/             # Session Memory
│   ├── guardrails/         # 安全与范围控制
│   └── observability/      # 日志与可观测性
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── memory/
├── Dockerfile.api
├── Dockerfile.streamlit
├── compose.yaml
└── requirements.txt
```

---

## 运行项目

创建环境变量文件：

```bash
cp .env.example .env
```

在 `.env` 中填写：

```env
OPENAI_API_KEY=replace_with_your_openai_api_key
OPENAI_MODEL=gpt-5-nano
TOOL_TIMEOUT_SECONDS=30
REQUEST_TIMEOUT_SECONDS=180
MAX_CONCURRENT_REQUESTS=4
```

Docker Compose 会自动为 Streamlit 配置后端服务地址，无需在 `.env` 中设置 `GROWTHGUARD_API_URL`。

> 公开仓库不包含业务数据。服务可以正常启动，但完整分析功能需要在本地提供兼容的数据文件。

使用 Docker 启动：

```bash
docker compose up --build
```

访问：

```text
Streamlit: http://localhost:8501
FastAPI Docs: http://localhost:8000/docs
```

---

## 数据说明

真实业务数据不会上传到公开仓库，包括：

```text
data/raw/
data/cleaned/
data/memory/
```

完整运行分析功能时，需要在本地提供与分析工具兼容的数据文件。

本仓库主要展示：

* AI Agent 架构设计
* 多工具调用
* 确定性数据分析
* 多轮对话与记忆
* Guardrails
* API 服务化
* 前端界面
* Docker 部署