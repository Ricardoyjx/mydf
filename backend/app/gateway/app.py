"""FastAPI 网关入口：应用生命周期管理 + 路由注册。"""

import os
import time
from pathlib import Path
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from my_df.agents.config.app_config import get_app_config
from app.gateway.routers.runs import router as runs_router
from app.gateway.routers.memory import router as memory_router
from app.gateway.routers.threads import router as threads_router
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
    app.include_router(memory_router)
    app.include_router(threads_router)

    cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 前端页面（SPA 单页入口）
    frontend_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend"

    @app.get("/")
    def get_root():
        return FileResponse(frontend_dir / "index.html")

    @app.get("/health", tags=["health"])
    def get_health():
        return {"status": "healthy", "service": "my-df-gateway"}

    # ── 请求日志中间件 ──
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """记录每个 HTTP 请求的方法、路径、状态码和处理耗时。"""
        start = time.perf_counter()
        method = request.method
        path = request.url.path
        query = request.url.query
        full_path = f"{path}?{query}" if query else path

        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start
            logger.error(
                "请求异常 | %s %s | %.0fms",
                method, full_path, elapsed * 1000,
                exc_info=True,
            )
            raise

        elapsed = time.perf_counter() - start
        # SSE 流式请求降级到 DEBUG，避免刷屏
        content_type = response.headers.get("content-type", "")
        log_fn = logger.debug if "text/event-stream" in content_type else logger.info
        log_fn(
            "%s %s | %s | %.0fms",
            method, full_path, response.status_code, elapsed * 1000,
        )
        return response

    return app


app = create_app()
