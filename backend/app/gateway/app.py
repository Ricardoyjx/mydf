"""FastAPI 网关入口：应用生命周期管理 + 路由注册。"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from my_df.agents.config.app_config import get_app_config
from app.gateway.routers.runs import router as runs_router
from app.gateway.config import get_gateway_config
from app.gateway.deps import langgraph_runtime

_BACKEND_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用程序生命周期处理器。"""

    # 1. 加载 .env 文件，使 DEEPSEEK_API_KEY 等环境变量可用
    env_path = _BACKEND_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        logger.info("已加载环境变量: %s", env_path)

    # 2. 加载应用配置（含模型、日志级别等）
    try:
        startup_config = get_app_config()
        logger.info("应用配置加载成功")
    except Exception as e:
        error_msg = f"网关启动时加载配置失败: {e}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    config = get_gateway_config()
    logger.info("API 网关启动于 %s:%s", config.host, config.port)

    # 3. 初始化 LangGraph 运行时（stream bridge、checkpointer 等）
    async with langgraph_runtime(app, startup_config):
        logger.info("LangGraph 运行时初始化完成")

        yield

    logger.info("API 网关关闭中")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""
    app = FastAPI(title="my-df Gateway", version="0.1.0", lifespan=lifespan)

    # API 路由
    app.include_router(runs_router)

    # 前端页面（SPA 单页入口）
    frontend_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend"

    @app.get("/")
    def get_root():
        return FileResponse(frontend_dir / "index.html")

    @app.get("/health", tags=["health"])
    def get_health():
        return {"status": "healthy", "service": "my-df-gateway"}

    return app


app = create_app()
