"""FastAPI 网关入口：应用生命周期管理 + 路由注册。"""

import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import ClassVar

from app.gateway.config import get_gateway_config
from app.gateway.deps import langgraph_runtime
from app.gateway.routers.memory import router as memory_router
from app.gateway.routers.runs import router as runs_router
from app.gateway.routers.threads import router as threads_router
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from my_df.config.app_config import get_app_config
from my_df.runtime.milvus.async_provider import make_milvus_storage

_BACKEND_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用程序生命周期处理器。"""

    # 0. 配置日志级别 + 彩色输出
    log_level = (os.getenv("MYDF_LOG_LEVEL") or "info").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    if not logging.getLogger().hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(_ColoredFormatter())
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

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

    # 3. 初始化 Milvus 向量存储（可选，失败不影响应用启动）
    try:
        async with make_milvus_storage() as milvus:
            await milvus.ensure_collection("default")
            app.state.milvus = milvus
            logger.info("Milvus 向量存储已就绪。")
    except Exception:  # noqa: BLE001
        logger.warning("Milvus 未就绪（向量存储不可用，其他功能正常）")
        app.state.milvus = None

    # 4. 初始化 LangGraph 运行时（stream bridge、checkpointer 等）
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
            logger.exception(
                "请求异常 | %s %s | %.0fms",
                method,
                full_path,
                elapsed * 1000,
                # exc_info=True,
            )
            raise

        elapsed = time.perf_counter() - start
        # SSE 流式请求降级到 DEBUG，避免刷屏
        content_type = response.headers.get("content-type", "")
        log_fn = logger.debug if "text/event-stream" in content_type else logger.info
        log_fn(
            "%s %s | %s | %.0fms",
            method,
            full_path,
            response.status_code,
            elapsed * 1000,
        )
        return response

    return app


class _ColoredFormatter(logging.Formatter):
    """带 ANSI 颜色的日志格式化器。"""

    _LEVEL_COLORS: ClassVar[dict] = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[41m",  # 红底
    }
    _RESET = "\033[0m"
    _BOLD = "\033[1m"
    _DIM = "\033[2m"

    def format(self, record: logging.LogRecord) -> str:
        level_color = self._LEVEL_COLORS.get(record.levelname, "")
        level = f"{level_color}{record.levelname}{self._RESET}"
        name = f"{self._DIM}{record.name}{self._RESET}"
        return f"{level} {name} | {record.getMessage()}"


app = create_app()
