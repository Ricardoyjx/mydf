from fastapi import FastAPI
from app.gateway.routers.runs import router as runs_router


def create_app() -> FastAPI:
    app = FastAPI()

    app.include_router(runs_router)

    @app.get("/health", tags=["health"])
    def get_health():
        return {"status": "health", "service": "my-df-gateway"}

    return app


app = create_app()
