# syntax=docker/dockerfile:1
# my-df 后端镜像：uv 管理依赖，单阶段构建（模型权重运行时下载并缓存到 volume）
#
# 构建上下文必须是项目根目录（需要 frontend/ 静态资源）：
#   docker build -f backend/Dockerfile -t my-df-app .

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/data/hf-cache

# uv 二进制（官方镜像）
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

WORKDIR /app

# 依赖清单先行，利用层缓存
COPY backend/pyproject.toml backend/uv.lock /app/
COPY backend/packages/harness/pyproject.toml /app/packages/harness/
COPY backend/packages/harness/my_df /app/packages/harness/my_df/

RUN uv sync --frozen --no-dev

# 应用代码
COPY backend/app /app/app

# 前端静态资源（后端服务托管，frontend_dir 解析到 /frontend）
COPY frontend /frontend

EXPOSE 8001

# 生产入口（无 reload；开发请用 start_dev.py）
CMD ["uv", "run", "uvicorn", "app.gateway.app:app", "--host", "0.0.0.0", "--port", "8001"]
