# GrowthGuard

GrowthGuard 是一个面向 DTC 与电商业务的 AI 增长分析 Agent。

用户可以直接使用自然语言提问，系统会自动解析业务意图、制定分析计划、调用确定性数据分析工具，并生成业务结论和行动建议。

项目基于 OpenAI Agents SDK 构建 Planner–Executor–Finalizer 工作流，并实现短期会话记忆与长期用户记忆双层机制。

> 本仓库不包含真实企业数据。原始数据与清洗后的生产数据因商业保密要求未公开。

---

## 核心功能

- 自然语言业务问答
- 自动分析规划与工具选择
- 销售、漏斗、订阅、留存、退款、营销和产品分析
- 多工具组合分析
- 多轮对话上下文解析
- 短期 Session Memory
- 长期 User Memory
- 跨会话用户偏好召回
- 结构化输出与 Guardrails
- 请求超时、并发控制和异常处理
- 结构化日志与请求追踪
- FastAPI 后端服务
- Streamlit 前端界面
- Docker Compose 容器化运行

---

## 系统架构

```text
用户问题
→ Streamlit
→ FastAPI
→ 短期 Session Memory
→ 长期 User Memory
→ Context Resolver
→ Planner
→ Deterministic Executor
→ Analytics Tools
→ Final Response Agent
→ 保存当前会话
→ 提取并更新长期记忆
→ 分析结论与行动建议
```

### 大语言模型负责

- 理解用户问题
- 解析多轮对话上下文
- 制定结构化分析计划
- 选择分析工具
- 提取长期用户偏好
- 解释工具返回的分析结果

### Python 分析工具负责

- 读取业务数据
- 计算业务指标
- 比较时间变化
- 返回结构化分析结果

指标计算与大语言模型解耦，业务数值仅由确定性 Python 工具生成，从而降低数字幻觉并提升结果的准确性和可追溯性。

---

## 双层记忆机制

GrowthGuard 使用两种不同作用范围的持久化记忆。

### 短期会话记忆

短期记忆通过 `session_id` 绑定当前对话，用于保存用户与 Agent 的历史消息，支持：

- 多轮连续追问
- 指代关系解析
- 页面刷新后的会话恢复
- 不同会话之间的数据隔离

例如：

```text
用户：最近订阅情况怎么样？
用户：那退款呢？
```

Agent 可以根据当前 Session 的历史记录理解第二个问题。

### 长期用户记忆

长期记忆通过 `user_id` 绑定用户，用于保存跨会话仍然有效的非敏感偏好，例如：

- 回答语言
- 回答风格
- 团队角色
- 重点关注方向
- 常看渠道
- 常看产品
- 长期业务目标

例如：

```text
以后默认用中文回答，先给结论。
我主要关注订阅留存和退款风险。
```

用户创建新会话后，Agent 仍可召回这些偏好。

长期记忆不会保存销售额、退款率、订阅人数等动态业务指标。此类数据始终由分析工具重新计算。

---

## 支持的分析模块

| 模块 | 功能 |
|---|---|
| Sales | 销售趋势与渠道贡献 |
| Funnel | 网站转化漏斗 |
| Subscription | 订阅增长、流失与停用 |
| Cohort | 客户留存分析 |
| Refund | 退款趋势与压力 |
| Marketing | Campaign 与 Flow 表现 |
| Product | 产品和 SKU 表现 |
| Multi-tool Analysis | 组合多个工具完成综合增长诊断 |

---

## 技术栈

- Python
- OpenAI Agents SDK
- FastAPI
- Streamlit
- Pydantic
- pandas
- SQLite
- Docker
- Docker Compose
- Pytest
- Structured Logging

---

## 项目结构

```text
growthguard/
├── api/
│   └── main.py                    # FastAPI 服务与记忆管理接口
├── app/
│   └── streamlit_app.py           # Streamlit 对话界面
├── sources/
│   ├── agent/
│   │   ├── agent_service.py       # Agent 主工作流
│   │   ├── context_resolver.py    # 多轮上下文解析
│   │   ├── memory_extractor.py    # 长期记忆提取
│   │   ├── planner.py             # 结构化分析规划
│   │   ├── executor.py            # 确定性工具执行
│   │   └── final_response_agent.py
│   ├── tools/                     # 七类数据分析工具
│   ├── memory/
│   │   ├── session_manager.py     # 短期会话记忆
│   │   └── long_term_memory.py    # 长期用户记忆
│   ├── guardrails/                # 输入、计划和输出校验
│   ├── core/                      # 异常与公共逻辑
│   └── observability/             # 日志与请求追踪
├── tests/
│   └── test_long_term_memory.py   # 长期记忆单元测试
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── memory/
├── evaluation/
├── Dockerfile.api
├── Dockerfile.streamlit
├── compose.yaml
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

---

## 运行项目

### 1. 创建环境变量

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

Docker Compose 会自动为 Streamlit 配置后端服务地址，不需要在 `.env` 中设置 `GROWTHGUARD_API_URL`。

### 2. 使用 Docker 启动

```bash
docker compose up --build
```

访问：

```text
Streamlit: http://localhost:8501
FastAPI Docs: http://localhost:8000/docs
```

### 3. 停止服务

```bash
docker compose down
```

---

## API 接口

| 方法 | 路径 | 功能 |
|---|---|---|
| `GET` | `/health` | 服务健康检查 |
| `POST` | `/ask` | 提交业务分析问题 |
| `GET` | `/sessions/{session_id}` | 查询短期会话记录 |
| `DELETE` | `/sessions/{session_id}` | 清除短期会话记录 |
| `GET` | `/users/{user_id}/memories` | 查询长期用户记忆 |
| `DELETE` | `/users/{user_id}/memories` | 清除长期用户记忆 |
| `GET` | `/evaluation/latest` | 查询最新评测结果 |

`POST /ask` 请求示例：

```json
{
  "question": "按我平时关注的方向，最近哪个问题更值得优先处理？",
  "session_id": "session_example_001",
  "user_id": "user_example_001"
}
```

---

## 测试

安装开发依赖：

```bash
python -m pip install -r requirements-dev.txt
```

运行全部测试：

```bash
python -m pytest -v
```

检查 Python 文件语法：

```bash
python -m compileall api app sources
```

长期记忆测试使用临时 SQLite 数据库，不会修改实际用户记忆文件

---

## 数据与隐私说明

公开仓库不包含以下内容：

```text
data/raw/
data/cleaned/
data/memory/
logs/
.env
```

其中：

- 真实业务数据不会上传到公开仓库
- API Key 不会上传到公开仓库
- Session Memory 数据库不会上传
- User Memory 数据库不会上传
- 日志文件不会上传

完整运行分析功能时，需要在本地提供与分析工具兼容的数据文件。

---

## 项目展示重点

本仓库主要展示：

- Planner–Executor–Finalizer Agent 工作流
- 结构化输出与工具路由
- 确定性业务指标计算
- 七类业务分析工具
- 长短期双层记忆机制
- 跨会话偏好提取与召回
- Guardrails 与异常处理
- FastAPI 服务化
- Streamlit 对话界面
- Docker 容器化部署
- Pytest 单元测试