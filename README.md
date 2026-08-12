# my-df

基于 LangGraph 的多 Agent 协作网关。FastAPI 提供 REST 与 SSE 流式接口，内置对话记忆（Memory）、知识库检索（RAG）、计划模式（Todo）等中间件；顶层 Supervisor 负责把任务委派给专业子代理，并通过质量评审闭环（Reflexion）保证输出质量。前端为单页 SPA。

## 技术栈

- Python 3.13，包管理统一使用 `uv`（禁止 pip / conda）
- FastAPI + Uvicorn（网关）
- LangGraph / LangChain（Agent 框架与中间件链）
- LangGraph Supervisor 编排（多 Agent 委派路由 + 子代理注册表 + 质量评审闭环）
- Milvus + sentence-transformers（向量检索 / RAG / 语义记忆）
- SQLite / PostgreSQL（checkpointer 持久化）

## 目录结构

```
my-df/
├── backend/
│   ├── app/gateway/                 # FastAPI 网关（路由、依赖注入、lifespan）
│   ├── packages/harness/            # my-df-harness：Agent 框架
│   │   └── my_df/
│   │       ├── agents/              # Supervisor 编排图、子代理、中间件、ThreadState
│   │       │   ├── supervisor_graph.py  # Supervisor 顶层编排图（路由委派 + 质量评审）
│   │       │   ├── sub_agent/           # 子代理（general-purpose / weather_search）
│   │       │   └── tools/               # 工具（search_weather 等）
│   │       ├── rag/                 # 知识库切块、入库、检索
│   │       ├── runtime/             # checkpointer / store / stream bridge / milvus
│   │       └── config/              # 环境变量与配置模型（含 subagent_config.py）
│   ├── tests/                       # pytest 测试
│   ├── docker-compose.yml           # Milvus / PostgreSQL 等基础设施
│   ├── start_dev.py                 # 开发启动脚本
│   └── pyproject.toml               # uv workspace
├── frontend/index.html              # 单页前端（由网关静态托管）
└── README.md
```

## 环境要求

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)（`curl -LsSf https://astral.sh/uv/install.sh | sh`）
- Docker（可选：启动 Milvus / PostgreSQL 时需要）

## 快速开始

### 1. 安装依赖

```bash
cd backend
uv sync
```

`uv sync` 会创建 `.venv` 并同时安装 workspace 内的 `my-df-harness`。

### 2. 配置环境变量

在 `backend/` 下创建 `.env`（模板如下，按需取消注释）。**注意：`.env` 文件中不要包含中文注释，防止跨平台加载编码错误。**

```bash
# LLM（必填，二选一）
MYDF_LLM_API_KEY=your_api_key
# DEEPSEEK_API_KEY=your_api_key

# 模型名称（默认 deepseek-v4-flash）
# MYDF_LLM_MODEL=deepseek-v4-flash

# 网关
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8001
GATEWAY_ENABLE_DOCS=true
MYDF_LOG_LEVEL=info

# 计划模式（TodoMiddleware，0/1）
# MYDF_IS_PLAN_MODE=1

# Checkpointer 持久化（默认 memory，重启后丢失）
# MYDF_CHECKPOINTER_TYPE=sqlite
# MYDF_CHECKPOINTER_PATH=.deer-flow/checkpoints.db

# PostgreSQL 持久化（配合 docker compose up -d postgres）
# MYDF_CHECKPOINTER_TYPE=postgres
# MYDF_CHECKPOINTER_PATH=postgresql://postgres:postgres@localhost:5432/mydf

# Embedding 模型（RAG / 语义记忆，默认 all-MiniLM-L6-v2）
# MYDF_EMBEDDING_MODEL=all-MiniLM-L6-v2
# HF_TOKEN=hf_xxx   # 下载 HuggingFace 模型需要

# Milvus（可选，默认 localhost:19530）
# MYDF_MILVUS_HOST=localhost
# MYDF_MILVUS_PORT=19530
# MYDF_MILVUS_DIM=384
# MYDF_MILVUS_INDEX_TYPE=IVF_FLAT
```

### 3. 启动基础设施（可选）

```bash
cd backend

# 完整 RAG 栈：etcd + minio + milvus + attu
docker compose up -d

# 仅 PostgreSQL（checkpointer 持久化）
docker compose up -d postgres
```

没有 Milvus / Embedding 模型时网关仍可启动，对话与记忆（JSON 存储）功能正常，RAG 与语义检索自动降级。

### 4. 启动网关

```bash
cd backend
.venv/bin/python start_dev.py
```

或先激活虚拟环境：

```bash
cd backend
source .venv/bin/activate
python start_dev.py
```

默认监听 `0.0.0.0:8001`，开发模式开启热重载（reload）。

### 5. 访问

- 前端对话页：<http://localhost:8001/>
- Swagger 文档：<http://localhost:8001/docs>
- 健康检查：<http://localhost:8001/health>

## 常用命令

```bash
cd backend

uv sync                        # 安装 / 更新依赖
uv run pytest                  # 运行测试
.venv/bin/python start_dev.py  # 启动开发服务器
uv pip install -e .            # 安装本地可编辑包
uv cache prune                 # 清理依赖缓存
```

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/runs/stream` | 创建运行，SSE 流式返回 Agent 输出 |
| GET | `/api/threads/{thread_id}/messages` | 查询线程对话历史 |
| GET | `/api/{thread_id}/memory` | 获取全局记忆 |
| POST | `/api/rag/documents` | 文本入库知识库 |
| POST | `/api/rag/documents/upload` | 文件内容入库知识库 |
| GET | `/api/rag/documents` | 知识库文档列表 |
| GET / POST | `/api/rag/search` | 知识库语义检索 |
| DELETE | `/api/rag/documents/{document_id}` | 删除知识库文档 |
| GET | `/health` | 健康检查 |

流式对话请求示例：

```bash
curl -N -X POST http://localhost:8001/api/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "lead_agent",
    "input": {"messages": [{"role": "user", "content": "你好"}]},
    "config": {"configurable": {"thread_id": "demo-thread"}}
  }'
```

## 核心机制

- **多 Agent 编排（Supervisor）**：顶层 LLM 绑定 `route_to_agent` 工具，按需把任务委派给注册表中的子代理；子代理在独立 `max_turns` / 超时约束下执行，结果回到共享消息流。
- **质量评审（Reflexion 闭环）**：子代理完成后先做零成本规则检查（空回答 / 回答过短），可选启用 LLM 评审（PASS/FAIL）；不通过时把评审反馈注入下一轮 Supervisor 重新编排，最多重试 `MAX_ROUTES=5` 次，超限强制结束，防止死循环。
- **中间件链**：Memory（持久化记忆 + Milvus 语义检索）→ RAG（知识库检索）→ DynamicContext（注入当前时间）→ Todo（`MYDF_IS_PLAN_MODE=1` 时启用）。
- **SSE 流式**：生产者（Agent worker）与消费者（SSE 端点）通过 `StreamBridge`（内存队列）解耦，支持心跳与 `Last-Event-ID` 断线重连。
- **持久化**：checkpointer 支持 `memory` / `sqlite` / `postgres`，与 store 共用同一配置段，保证重启后对话历史不丢失。

## 多 Agent 架构

### 编排流程

```mermaid
flowchart TD
    START --> model_call[model_call<br/>Supervisor LLM<br/>绑定 route_to_agent 工具]
    model_call --> supervisor[supervisor<br/>解析委派意图]
    supervisor -->|合法委派| sub_agent[sub_agent<br/>子代理独立执行]
    supervisor -->|直接回答 / 非法目标| END
    sub_agent --> reflect[reflect<br/>质量评审]
    reflect -->|通过| END
    reflect -->|不通过且未超限| model_call
    reflect -->|超过 MAX_ROUTES| END
```

### 子代理注册表

每个子代理由 `SubagentConfig`（`my_df/config/subagent_config.py`）定义：

| 字段 | 说明 |
| --- | --- |
| `name` / `description` | 名称与能力描述；description 会注入 Supervisor 提示词，供其决策委派 |
| `system_prompt` | 子代理系统提示词 |
| `tools` / `disallowed_tools` | 工具白名单 / 黑名单过滤 |
| `model` | 模型名；`"inherit"` 表示复用默认模型 |
| `max_turns` / `timeout_seconds` | 子代理递归上限与执行超时，防止子代理失控 |

内置子代理：

- `general-purpose`：通用助手，继承全部工具，负责多步复杂任务，输出结构化总结。
- `weather_search`：天气查询子代理，调用 `search_weather` 工具（wttr.in 免费接口，无需 API key），英文天气描述自动翻译为中文。

新增子代理三步：在 `agents/sub_agent/` 实现子代理（复用 `make_assistant_subagent` 或自定义节点）→ 定义 `SubagentConfig` → 在 `supervisor_graph._build_default_registry` 中注册。

### 运行层适配

- worker 递归提取子代理嵌套输出，并按消息 id 去重，避免同一消息重复推送。
- token 消耗按 `lead_agent`（supervisor / reflect）与 `subagent` 分账，便于成本观测。
- 网关启动时预热 Supervisor 编排图，请求复用同一实例；构建失败自动降级为按需构建。

## 常见问题

**`ModuleNotFoundError: No module named 'uvicorn'`**

使用了系统 Python 而不是项目虚拟环境。请使用 `.venv/bin/python start_dev.py`，或先 `source .venv/bin/activate`。

**启动日志出现 "Milvus 未就绪" / "Embedding 模型加载失败"**

说明向量基础设施不可用，不影响基础对话。如需 RAG 与语义记忆，先执行 `docker compose up -d` 并确认 `.env` 中相关配置。

**模型下载慢或失败**

在 `.env` 中配置 `HF_TOKEN`，或将 `MYDF_EMBEDDING_MODEL` 换成更小的模型。
