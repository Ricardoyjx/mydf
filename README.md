# my-df

基于 LangGraph 的多 Agent 协作网关。FastAPI 提供 REST 与 SSE 流式接口，内置对话记忆（Memory）、知识库检索（RAG）、计划模式（Todo）等中间件，前端为单页 SPA。

## 技术栈

- Python 3.13，包管理统一使用 `uv`（禁止 pip / conda）
- FastAPI + Uvicorn（网关）
- LangGraph / LangChain（Agent 框架与中间件链）
- Milvus + sentence-transformers（向量检索 / RAG / 语义记忆）
- SQLite / PostgreSQL（checkpointer 持久化）

## 目录结构

```
my-df/
├── backend/
│   ├── app/gateway/                 # FastAPI 网关（路由、依赖注入、lifespan）
│   ├── packages/harness/            # my-df-harness：Agent 框架
│   │   └── my_df/
│   │       ├── agents/              # Lead Agent、中间件、ThreadState
│   │       ├── rag/                 # 知识库切块、入库、检索
│   │       ├── runtime/             # checkpointer / store / stream bridge / milvus
│   │       └── config/              # 环境变量与配置模型
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

- **中间件链**：Memory（持久化记忆 + Milvus 语义检索）→ RAG（知识库检索）→ DynamicContext（注入当前时间）→ Todo（`MYDF_IS_PLAN_MODE=1` 时启用）。
- **SSE 流式**：生产者（Agent worker）与消费者（SSE 端点）通过 `StreamBridge`（内存队列）解耦，支持心跳与 `Last-Event-ID` 断线重连。
- **持久化**：checkpointer 支持 `memory` / `sqlite` / `postgres`，与 store 共用同一配置段，保证重启后对话历史不丢失。

## 常见问题

**`ModuleNotFoundError: No module named 'uvicorn'`**

使用了系统 Python 而不是项目虚拟环境。请使用 `.venv/bin/python start_dev.py`，或先 `source .venv/bin/activate`。

**启动日志出现 "Milvus 未就绪" / "Embedding 模型加载失败"**

说明向量基础设施不可用，不影响基础对话。如需 RAG 与语义记忆，先执行 `docker compose up -d` 并确认 `.env` 中相关配置。

**模型下载慢或失败**

在 `.env` 中配置 `HF_TOKEN`，或将 `MYDF_EMBEDDING_MODEL` 换成更小的模型。
