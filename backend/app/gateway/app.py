from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from my_df.agents.config.app_config import get_app_config
from app.gateway.routers.runs import router as runs_router
from app.gateway.config import get_gateway_config
from app.gateway.deps import langgraph_runtime

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""

    # Load config and check necessary environment variables at startup.
    # `startup_config` is a local snapshot used only for one-shot bootstrap
    # work (logging level, langgraph_runtime engines, channels). Request-time
    # config resolution always routes through `get_app_config()` in
    # `app/gateway/deps.py::get_config()` so `config.yaml` edits become
    # visible without a process restart. We deliberately do NOT cache this
    # snapshot on `app.state` to keep that contract enforceable.
    try:
        startup_config = get_app_config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        error_msg = f"Failed to load configuration during gateway startup: {e}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    config = get_gateway_config()
    logger.info(f"Starting API Gateway on {config.host}:{config.port}")

    async with langgraph_runtime(app, startup_config):
        logger.info("LangGraph runtime initialised")

        yield

    logger.info("Shutting down API Gateway")


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    app.include_router(runs_router)

    # 挂载静态文件，提供前端交互页面
    # frontend/ 目录位于项目根目录
    frontend_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend"
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    @app.get("/health", tags=["health"])
    def get_health():
        return {"status": "health", "service": "my-df-gateway"}

    return app


app = create_app()
