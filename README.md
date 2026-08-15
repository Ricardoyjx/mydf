# my-df

基于 LangGraph 的多 Agent 协作网关。FastAPI 提供 REST 与 SSE 流式接口，顶层 Supervisor 负责把任务委派给专业子代理，并通过质量评审闭环（Reflexion）保证输出质量；内置 RAG 知识库、分层记忆、运行管理 API、可观测性事件流水与评测驱动开发（EDD）体系。前端为单页 SPA（对话 / 知识库 / 记忆 / 观测）。

## 技术栈

- Python 3.13，包管理统一使用 `uv`（禁止 pip / conda）
- FastAPI + Uvicorn（网关）
- LangGraph / LangChain（Agent 框架与图编排）
- Supervisor 多 Agent 编排（子代理注册表 + 路由委派 + Reflexion 质量评审闭环）
- Milvus + sentence-transformers（向量检索 / RAG / 语义记忆 / rerank 精排）
- PostgreSQL（checkpointer / RunStore / 事件流水持久化）
- Docker Compose（基础设施 + 应用一键编排）

## 目录结构

```bash
my-df/
├── backend/
│   ├── app/gateway/                 # FastAPI 网关（路由、依赖注入、lifespan）
│   ├── packages/harness/my_df/
│   │   ├── agents/
│   │   │   ├── supervisor_graph.py      # Supervisor 编排图（路由委派 + 质量评审 + 上下文注入）
│   │   │   ├── sub_agent/               # 子代理工厂（general-purpose / weather_search）
│   │   │   ├── tools/                   # 工具（search_weather 等）
│   │   │   ├── middlewares/             # 记忆 / RAG / 动态上下文中间件（supervisor 节点复用）
│   │   │   └── thread_state.py          # 线程状态（sandbox / route_count / 评审反馈）
│   │   ├── sandbox/                 # 沙箱（LocalSandbox / Provider 注册发现 / 单例）
│   │   ├── rag/                     # 知识库切块、入库、检索
│   │   ├── runtime/                 # checkpointer / store / stream bridge / runs / events / milvus
│   │   └── config/                  # 环境变量与配置模型（subagent / sandbox 等）
│   ├── evals/                       # 评测工程（datasets / evaluators / configs / reports）
│   ├── scripts/eval_agents.py       # 评测脚本（路由 / 工具 / 内容三维判定）
│   ├── tests/                       # pytest 测试（82 个）
│   ├── Dockerfile                   # 应用镜像（uv 构建，模型懒加载）
│   ├── docker-compose.yml           # 基础设施 + app 编排
│   ├── start_dev.py                 # 开发启动脚本（无热重载）
│   └── pyproject.toml               # uv workspace
├── frontend/index.html              # 单页前端（对话 / 知识库 / 记忆 / 观测）
└── README.md
```

## 环境要求

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)（`curl -LsSf https://astral.sh/uv/install.sh | sh`）
- Docker（可选：启动 Milvus / PostgreSQL / 应用编排时需要）

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

# 计划模式（TodoMiddleware，1 开启）
# MYDF_IS_PLAN_MODE=0
# MYDF_DEBUG=false

# 网关
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8001
GATEWAY_ENABLE_DOCS=true
MYDF_LOG_LEVEL=info

# PostgreSQL 持久化（checkpointer + 事件流水，推荐）
MYDF_CHECKPOINTER_TYPE=postgres
MYDF_CHECKPOINTER_PATH=postgresql://postgres:postgres@localhost:5432/mydf

# 轻量备选：SQLite（单文件，无需 Docker）
# MYDF_CHECKPOINTER_TYPE=sqlite
# MYDF_CHECKPOINTER_PATH=.my_df/checkpoints.db

# Embedding 模型（RAG / 语义记忆，懒加载：首次检索时下载）
# MYDF_EMBEDDING_MODEL=all-MiniLM-L6-v2
# HF_TOKEN=hf_xxx

# Reranker 精排（懒加载：首次搜索时加载，MYDF_RERANK_ENABLED=true 开启）
# MYDF_RERANK_ENABLED=true
# MYDF_RERANK_MODEL=BAAI/bge-reranker-base

# Milvus（可选，默认 localhost:19530）
# MYDF_MILVUS_HOST=localhost
# MYDF_MILVUS_PORT=19530
# MYDF_MILVUS_DIM=384
# MYDF_MILVUS_INDEX_TYPE=IVF_FLAT

# 沙箱 Provider（默认本地沙箱，可替换为任意已注册类路径）
# MYDF_SANDBOX_PROVIDER=my_df.sandbox.local_sandbox_provider.LocalSandboxProvider
```

#### Windows（WSL2）环境注意事项

1. **编码与换行**：`.env` 必须为 **UTF-8 无 BOM**，换行符建议用 **LF**（VS Code 右下角可切换；Windows 记事本保存易带 BOM/CRLF，可能引发解析异常）。
2. **localhost 语义**：服务运行在 WSL2 内，`localhost:5432` / `localhost:19530` 指向 WSL 内 Docker 映射的 PG / Milvus——正常直连即可。**若 PG / Milvus 跑在 Windows 原生侧**，需把 DSN / host 改为 Windows 宿主 IP（`ipconfig` 查以太网 IPv4，如 `192.168.x.x`）。
3. **局域网访问**：Windows 浏览器访问 `localhost:8001` 走 WSL2 自动转发，无需额外配置；局域网其他设备访问需 `netsh interface portproxy` 指向 WSL2 当前 IP（详见"常见问题"）。
4. **sqlite 路径**：使用 `MYDF_CHECKPOINTER_PATH` 时建议填相对路径（如 `.my_df/checkpoints.db`），避免 Windows 风格反斜杠在 DSN 解析时被转义。

### 3. 启动基础设施（可选）

```bash
cd backend

# 完整栈：etcd + minio + milvus + attu + postgres + app
docker compose up -d

# 仅 PostgreSQL（checkpointer / 事件持久化）
docker compose up -d postgres
```

没有 Milvus / Embedding 时网关仍可启动，对话与记忆功能正常，RAG 与语义检索自动降级（Milvus 连接快速失败，约 5 秒）。

### 4. 启动网关

```bash
cd backend
.venv/bin/python start_dev.py
```

默认监听 `0.0.0.0:8001`。**开发模式关闭热重载**——改动代码后需手动重启生效。

### 5. 访问

- 前端（对话 / 知识库 / 记忆 / 观测）：<http://localhost:8001/>
- Swagger 文档：<http://localhost:8001/docs>
- 健康检查：<http://localhost:8001/health>

## 常用命令

```bash
cd backend

uv sync                        # 安装 / 更新依赖
uv run pytest                  # 运行测试（82 个）
.venv/bin/python start_dev.py  # 启动开发服务器
.venv/bin/python scripts/eval_agents.py   # 运行 Agent 评测（EDD）
docker compose build app       # 构建应用镜像
docker compose up -d app       # 一键启动（含依赖）
```

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/runs/stream` | 创建运行，SSE 流式返回 Agent 输出 |
| GET | `/api/runs` | 运行列表（状态 / token / 线程过滤） |
| GET | `/api/runs/{run_id}/events` | 运行事件流水（可观测性：route/subagent/reflect/token） |
| DELETE | `/api/runs/{run_id}` | 删除运行记录 |
| POST | `/api/runs/{run_id}/cancel` | 取消运行 |
| GET | `/api/threads/{thread_id}/messages` | 查询线程对话历史 |
| GET | `/api/threads/{thread_id}/runs` | 线程运行历史 |
| GET | `/api/threads/{thread_id}/runs/usage` | 线程 token 用量（lead/subagent/middleware 分账） |
| GET | `/api/{thread_id}/memory` | 获取全局记忆 |
| POST | `/api/rag/documents` | 文本入库知识库 |
| POST | `/api/rag/documents/upload` | 文件内容入库知识库 |
| GET | `/api/rag/documents` | 知识库文档列表 |
| GET / POST | `/api/rag/search` | 知识库语义检索（含 rerank 精排） |
| DELETE | `/api/rag/documents/{document_id}` | 删除知识库文档 |
| GET | `/health` | 健康检查 |

流式对话请求示例：

```bash
curl -N -X POST http://localhost:8001/api/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "lead_agent",
    "input": {"messages": [{"role": "user", "content": "武汉明天天气怎么样？"}]},
    "config": {"configurable": {"thread_id": "demo-thread"}}
  }'
```

## 多 Agent 架构

### 编排流程

```mermaid
flowchart TD
    START --> model_call[model_call<br/>Supervisor LLM<br/>注入记忆/RAG + 绑定 route_to_agent]
    model_call --> supervisor[supervisor<br/>解析委派意图]
    supervisor -->|合法委派| sub_agent[sub_agent<br/>子代理独立执行]
    supervisor -->|直接回答 / 非法目标| END
    sub_agent --> reflect[reflect<br/>规则 + LLM 质量评审]
    reflect -->|通过| END
    reflect -->|不通过且未超限| model_call
    reflect -->|超过 MAX_ROUTES| END
```

### 子代理注册表

每个子代理由 `SubagentConfig`（`my_df/config/subagent_config.py`）定义，注册在 `supervisor_graph._build_default_registry`：

| 字段 | 说明 |
| --- | --- |
| `name` / `description` | 名称与能力描述；description 注入 Supervisor 提示词供其决策 |
| `system_prompt` | 子代理系统提示词 |
| `tools` / `disallowed_tools` | 工具白名单 / 黑名单过滤 |
| `model` | 模型名；`"inherit"` 复用默认模型（全图共享单实例） |
| `max_turns` / `timeout_seconds` | 递归上限（映射 recursion_limit）与执行超时 |

内置子代理：

- `general-purpose`：通用助手，多步复杂任务；
- `weather_search`：天气查询，调用 `search_weather`（wttr.in 免费接口，无需 key），英文描述自动翻译为中文。

新增子代理三步：实现子代理工厂 → 定义 `SubagentConfig` → 注册到 `_build_default_registry`（description 写清适用/不适用场景，决定委派准确性）。

### 质量评审（Reflexion 闭环）

- 零成本规则检查（空回答 / 过短）+ 可选 LLM 评审（PASS/FAIL + 一句反馈）；
- 不通过时反馈注入下一轮 Supervisor（要求：数据缺失必须重新委派，不得直接结束）；
- `route_count` 覆盖式 reducer，每次运行从 0 开始（不跨 run 累计），超过 `MAX_ROUTES=5` 强制结束并标记 error；
- 孤儿 tool_calls 消息自动清洗，防止历史数据触发模型 API 400。

### 记忆 / RAG 注入

Supervisor 的 `model_call` 节点复用 MemoryMiddleware / RagMiddleware 的注入逻辑（读记忆 + 知识库检索），模型调用后回写对话摘要与事实——与旧 `create_agent` 中间件链行为一致。embedding 与 reranker 均为懒加载，首次使用才加载。

## 可观测性

每次运行的事件流（`run_start → token → route → subagent_start/end → reflect → run_end`）写入 PG 事件表，前端"观测"页按时间线可视化，支持事件类型过滤：

- 运行状态 / token 分账（lead / subagent）；
- 委派决策（目标子代理、任务、是否重试轮）；
- 质量评审结果（passed / feedback / 轮次 / 是否超限）。

```bash
curl "http://localhost:8001/api/runs/{run_id}/events"
```

## 评测驱动开发（EDD）

`evals/` 标准评测工程：

```bash
backend/evals/
├── configs/eval_config.yaml    # 评测配置（LLM 评审开关 / 超时 / recursion_limit）
├── datasets/cases.yaml         # 场景集（input + 期望 route + must_contain）
├── evaluators/                 # 判定器（route / tool / content）
└── reports/                    # 评测报告产物
```

运行：

```bash
cd backend && .venv/bin/python scripts/eval_agents.py
```

每个场景三维判定（路由正确性 / 工具调用 / 内容关键词），输出通过率与失败原因——改 prompt / 图 / 子代理后跑一遍对比，即评估驱动开发迭代循环。

## 沙箱

本地沙箱模块（`my_df/sandbox/`）：

- `Sandbox` ABC：execute_command / read_file / write_file / list_dir；
- `LocalSandbox`：容器路径 ↔ 本地路径映射、越界防护、只读标记；
- `LocalSandboxProvider`：线程级沙箱（`local:{thread_id}`）+ LRU 缓存 + 按线程目录隔离；
- Provider 注册与发现：内置别名 / `register_provider()` 自定义 / 完整类路径动态解析（`importlib`），配置项 `MYDF_SANDBOX_PROVIDER`。

## 常见问题

**`ModuleNotFoundError: No module named 'uvicorn'`**

使用了系统 Python 而不是项目虚拟环境。请使用 `.venv/bin/python start_dev.py`，或先 `source .venv/bin/activate`。

**启动日志出现 "Milvus 未就绪" / "Embedding 模型加载失败"**

向量基础设施不可用（Milvus 连接快速失败约 5 秒），不影响基础对话。如需 RAG 与语义记忆，先 `docker compose up -d` 并确认 `.env` 配置。

**模型下载慢或失败**

在 `.env` 配置 `HF_TOKEN`，或换更小的 embedding 模型。embedding / reranker 均为懒加载，首次检索时下载（HF 缓存挂载在容器 volume）。

**开发时改代码不生效**

`start_dev.py` 已关闭热重载（reload=False），改动后需手动重启服务。

**Windows 无法访问 `localhost:8001`（连接被重置）**

通常是 WSL2 的 netsh portproxy 规则指向了旧 WSL2 IP。删除 `0.0.0.0:8001` 的 portproxy 规则（`netsh interface portproxy delete v4tov4 listenport=8001 listenaddress=0.0.0.0`），改由 WSL2 自动 localhost 转发（动态适配 IP）。
