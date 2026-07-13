import os as _os

from fastapi import FastAPI
from my_df.runtime.runs.schema import DisconnectMode
from app.gateway.routers.runs import router as runs_router


def create_app() -> FastAPI:
    app = FastAPI()

    app.include_router(runs_router)

    if _os.getenv("MYDF_DEBUG"):
        _setup_debug(app)

    @app.get("/health", tags=["health"])
    def get_health():
        return {"status": "health", "service": "my-df-gateway"}

    return app


def _setup_debug(app: FastAPI) -> None:
    """Debug mode: 用 InMemoryStreamBridge + mock LLM 真实驱动 middleware。"""
    import asyncio
    from datetime import datetime
    import uuid

    from unittest.mock import MagicMock

    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage

    from my_df.runtime.stream_bridge.memory import InMemoryStreamBridge
    from my_df.runtime.runs.manager import RunManager
    from my_df.runtime.runs.schema import RunStatus
    from my_df.agents.middlewares.todo_middleware import TodoMiddleware
    from my_df.agents.thread_state import ThreadState

    bridge = InMemoryStreamBridge()
    run_mgr = RunManager()
    app.state.stream_bridge = bridge
    app.state.run_manager = run_mgr
    print("[debug] InMemoryStreamBridge + RunManager 已注入")

    # ── mock LLM: 连续两次回复，演示 write_todos 行为 ──
    model = MagicMock()
    _call_count = [0]  # 用 list 绕过 nonlocal

    async def _mock_ainvoke(*_a, **_kw):
        n = _call_count[0]
        _call_count[0] += 1

        if n == 0:
            return AIMessage(
                content="我来帮你完成。先列个计划。",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {
                            "todos": [
                                {"content": "分析需求", "status": "in_progress"},
                                {"content": "编写代码", "status": "pending"},
                                {"content": "运行测试", "status": "pending"},
                            ]
                        },
                        "id": "call_todos_1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            return AIMessage(
                content="已标记完成，继续。",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {
                            "todos": [
                                {"content": "分析需求", "status": "completed"},
                                {"content": "编写代码", "status": "in_progress"},
                                {"content": "运行测试", "status": "pending"},
                            ]
                        },
                        "id": "call_todos_2",
                        "type": "tool_call",
                    }
                ],
            )

    model.ainvoke = _mock_ainvoke

    agent = create_agent(
        model=model,
        middleware=[
            TodoMiddleware(
                system_prompt="你是一个开发助手，使用 write_todos 管理任务。",
                tool_description="创建和管理待办事项列表。",
            )
        ],
        state_schema=ThreadState,
    )

    # ── 替换 start_run（同时更新 services 和 runs 两个模块的引用）──
    import app.gateway.services as svc
    import app.gateway.routers.runs as rtr

    async def _debug_start_run(body, thread_id: str, request):
        run_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        from my_df.runtime.runs.manager import RunRecord

        record = RunRecord(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id=getattr(body, "assistant_id", None),
            status=RunStatus.pending,
            on_disconnect=DisconnectMode.cancel,
            created_at=now,
            updated_at=now,
        )
        run_mgr._runs[run_id] = record

        await bridge.publish(run_id, "metadata", {"run_id": run_id})

        inp = body.input if hasattr(body, "input") and body.input else {"messages": [{"role": "user", "content": "帮我写个程序"}]}  # fmt: skip

        async def _run():
            try:
                async for chunk in agent.astream(
                    inp,
                    {"recursion_limit": 100, "configurable": {"run_id": run_id}},
                ):
                    await bridge.publish(run_id, "updates", {"agent": chunk})
            except Exception as exc:
                await bridge.publish(run_id, "error", {"message": str(exc)})
            finally:
                record.status = RunStatus.success
                await bridge.publish_end(run_id)

        record.task = asyncio.create_task(_run())
        record.status = RunStatus.running
        return record

    # 同时替换两个模块的引用
    svc.start_run = _debug_start_run
    rtr.start_run = _debug_start_run
    print("[debug] services.start_run & routers.runs.start_run 已替换")


app = create_app()
