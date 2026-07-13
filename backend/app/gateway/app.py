import os as _os

from fastapi import FastAPI
from app.gateway.routers.runs import router as runs_router


def create_app() -> FastAPI:
    app = FastAPI()

    app.include_router(runs_router)

    # Debug mode: 注入 mock 依赖，方便本地 curl 测试
    if _os.getenv("MYDF_DEBUG"):
        from unittest.mock import AsyncMock

        app.state.stream_bridge = AsyncMock()
        app.state.run_manager = AsyncMock()
        print("[debug] mock stream_bridge & run_manager injected")

    @app.get("/health", tags=["health"])
    def get_health():
        return {"status": "health", "service": "my-df-gateway"}

    return app


app = create_app()
